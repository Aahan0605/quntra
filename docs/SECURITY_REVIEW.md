# QuNtra security review — 2026-07-28

Scope: secret handling, network exposure, authentication, deserialisation,
injection, data durability. Findings ordered by severity.

---

## CRITICAL

### S1. Telegram bot token was written to logs in cleartext, 13,143 times
`logs/telegram_bot.log` (11 MB, mode 644) contained the live token in every
httpx request URL, because the Telegram API places the token in the URL
path and httpx logs full URLs at INFO.

Anything able to read that file — any process running as the user, any
backup, any log-shipping tool, any screen share — had full control of the
bot: read every alert, issue every command, including `/start_trading`.

- **Fixed:** `src/utils/log_redaction.py` scrubs token shapes and known
  secret values from every record, and turns httpx/httpcore down to WARNING.
  Existing logs scrubbed; all logs and `config/secrets.env` chmod 600.
- **Outstanding — operator action:** *the token itself is still burned.*
  Revoke via @BotFather and run `scripts/rotate_telegram_token.py`.
  Redaction prevents recurrence; it does not un-leak what already leaked.

### S2. PostgreSQL published on all interfaces with a documented password
The container binds `0.0.0.0:5432` with `POSTGRES_PASSWORD=quntra_dev` — a
value printed verbatim in `RUNBOOK.md`, which is committed to the repo.

Any device on the same network can read and write the trade database:
signals, positions, P&L, research. On untrusted wifi this is remote
takeover of the trading record.

**Fix (not applied — needs a maintenance window and your sign-off):**
```bash
docker stop quntra-db
docker run -d --name quntra-db-new --restart unless-stopped \
  -e POSTGRES_USER=quntra -e POSTGRES_PASSWORD="$(openssl rand -base64 24)" \
  -e POSTGRES_DB=quntra -v quntra-pgdata:/var/lib/postgresql/data \
  -p 127.0.0.1:5432:5432 postgres:15
# then restore data/backups/quntra_full_20260728.sql and update POSTGRES_URL
```
This also fixes S4. Backup already taken: `data/backups/quntra_full_20260728.sql` (3.0 MB).

---

## HIGH

### S3. Unauthenticated API bound to every interface
`api/main.py` ran uvicorn on `0.0.0.0:8000` with `allow_origins=["*"]` plus
`allow_credentials=True`, and **no authentication on any router** —
portfolio, ML, backtest, options all anonymous.

- **Fixed:** binds `127.0.0.1` by default (override via `QUNTRA_API_HOST`);
  CORS narrowed to explicit local origins and GET/POST.
- **Outstanding:** the routers still have no auth. Loopback binding is the
  control; do not expose this service without adding one.

### S4. Trade history lives on an anonymous Docker volume
The volume is unnamed (`e13db0c3a134…`). It survives `docker rm`, but
`docker volume prune` — a routine cleanup command — destroys every trade,
signal and research note with no warning and no backup.

**Fix:** the named-volume migration in S2. Until then, `pg_dump` on a
schedule. There is currently no automated backup at all.

---

## MEDIUM

### S5. Bot ownership is re-claimable if `system_state` is ever emptied
`is_authorized()` treats an empty whitelist as "unclaimed", so the next
message received claims the bot (`telegram_bot.py:281-316`).

Sensible for bootstrapping, dangerous afterwards: any event that clears
`system_state` — DB restore from an early dump, the S4 volume loss, a
migration — silently reopens enrolment. Combined with S1, where the token
was public for weeks, a stranger messaging first would take the bot.

**Recommended:** once `TELEGRAM_CHAT_ID` is set in `config/secrets.env`,
treat it as the authority and refuse first-contact enrolment. Re-enrolment
should require an explicit CLI action, not merely an empty table.

### S6. `pickle.load()` on model artefacts
Four call sites (`council.py:316`, `daily_trainer.py:180`,
`overnight_pipeline.py:187`, `verify_models.py:52`). Pickle executes
arbitrary code on load. Files are locally generated, so this is not
currently exploitable — but anything with write access to `data/models/`
achieves code execution inside the trading process.

**Recommended:** `chmod 700 data/models*`; longer term prefer
`xgboost.Booster.save_model` (JSON) over pickle.

### S7. `shell=True` throughout the setup script
`scripts/complete_local_setup.py:55` runs every command through the shell.
Commands are literals today, so there is no injection path — but the helper
invites one, and it is invoked with operator privileges.

**Recommended:** pass argv lists; drop `shell=True`.

---

## LOW

### S8. No rate limiting on Telegram commands
An authorised chat can issue unlimited commands. Some (`/chat`, research
triggers) call paid APIs. Low impact given single-operator use; would matter
if the whitelist ever grew.

### S9. Secrets loaded into `os.environ` process-wide
Any dependency can read `KITE_API_SECRET` via `os.environ`. Standard
practice, noted for completeness.

---

## Verified as sound

- `config/secrets.env` and `.env` are correctly gitignored; no secret is
  committed. `git grep` over tracked files finds no credentials.
- No SQL string interpolation anywhere — SQLAlchemy ORM throughout, so no
  injection surface.
- No `eval`/`exec` on external input (the `.eval()` hits are PyTorch
  inference-mode calls).
- Unauthorised Telegram contacts are rejected *silently*, not told the bot
  exists — correct choice.
- The Docker restart policy is `unless-stopped`, so the DB returns after a
  reboot.

---

## Priority order

1. **Rotate the Telegram token** (S1) — only you can do this
2. Recreate Postgres: localhost binding, strong password, named volume (S2 + S4)
3. Close bot re-enrolment once the chat ID is known (S5)
4. Add authentication before the API is ever exposed (S3)
5. Tighten `data/models` permissions (S6)

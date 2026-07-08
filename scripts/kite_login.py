#!/usr/bin/env python3
"""
Kite (Zerodha) daily login helper — generates today's ACCESS TOKEN.

WHY THIS EXISTS: Kite access tokens are NOT permanent. They expire every
morning (~07:30 IST), so a fresh one must be generated each trading day
from your permanent API key + secret via a one-time login. This script
does that and writes KITE_ACCESS_TOKEN into config/secrets.env.

⚠️  This does NOT enable live trading. Live capital stays locked until the
    40-day paper gate passes (see STATUS.md). This only prepares the daily
    token so that, WHEN the gate passes, the switch is one step.

USAGE (run each trading morning, once live):
  1. Put your permanent creds in config/secrets.env:
       KITE_API_KEY=...      KITE_API_SECRET=...
  2. Print the login URL:
       python3 scripts/kite_login.py --login-url
  3. Open it, log in, approve. You land on a redirect URL containing
     ?request_token=XXXX  — copy that value.
  4. Exchange it for today's access token:
       python3 scripts/kite_login.py --request-token XXXX
     The access token is written to config/secrets.env.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENV = ROOT / "config" / "secrets.env"


def _read_env() -> dict:
    vals: dict[str, str] = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip()
    return vals


def _write_env(key: str, value: str) -> None:
    lines = ENV.read_text().splitlines() if ENV.exists() else []
    lines = [l for l in lines if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    ENV.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--login-url", action="store_true",
                    help="print the Kite login URL and exit")
    ap.add_argument("--request-token",
                    help="request_token from the redirect URL after login")
    args = ap.parse_args()

    env = _read_env()
    api_key = env.get("KITE_API_KEY", "")
    api_secret = env.get("KITE_API_SECRET", "")
    if not api_key:
        print("ERROR: KITE_API_KEY is empty in config/secrets.env — add your "
              "permanent Zerodha API key first (from developers.kite.trade).")
        return 1

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("ERROR: kiteconnect not installed. pip install kiteconnect")
        return 1

    kite = KiteConnect(api_key=api_key)

    if args.login_url:
        print("\nOpen this URL, log in, and approve:")
        print("  " + kite.login_url())
        print("\nThen copy the request_token from the redirect URL and run:")
        print("  python3 scripts/kite_login.py --request-token <TOKEN>")
        return 0

    if args.request_token:
        if not api_secret:
            print("ERROR: KITE_API_SECRET is empty — needed to exchange the "
                  "request_token.")
            return 1
        try:
            data = kite.generate_session(args.request_token,
                                         api_secret=api_secret)
            access_token = data["access_token"]
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: token exchange failed: {e}")
            print("(request_token is single-use and expires in minutes — "
                  "get a fresh one from the login URL.)")
            return 1
        _write_env("KITE_ACCESS_TOKEN", access_token)
        print(f"✓ Access token written to config/secrets.env "
              f"({access_token[:6]}…). Valid until ~07:30 IST tomorrow.")
        print("Live trading still requires the paper gate to have passed and "
              "HUMAN_APPROVAL_REQUIRED=true — this only prepares the token.")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

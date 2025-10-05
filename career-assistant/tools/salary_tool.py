# tools/salary_tool.py
# Purpose: Salary lookup via local CSV file (role,state,annual_aud) with free-text parsing.
import os, re
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

AU_STATE_ALIASES = {
    "nsw": "NSW", "new south wales": "NSW",
    "vic": "VIC", "victoria": "VIC",
    "qld": "QLD", "queensland": "QLD",
    "sa": "SA", "south australia": "SA",
    "wa": "WA", "western australia": "WA",
    "tas": "TAS", "tasmania": "TAS",
    "act": "ACT", "australian capital territory": "ACT",
    "nt": "NT", "northern territory": "NT",
    "australia": "AU",
}

def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _extract_role_and_state(q: str, known_roles) -> Tuple[str, Optional[str]]:
    """Extract role and optional state from a free-text query."""
    qn = _norm(q)

    # detect state
    state = None
    for k, v in AU_STATE_ALIASES.items():
        if k in qn:
            state = v if v != "AU" else None
            break

    # exact role match
    for r in known_roles:
        if _norm(r) in qn:
            return r, state

    # fallback guess
    return q.strip(), state

class SalaryTool:
    def __init__(self, backend: str = "csv", country: str = "au", csv_path: str = "data/salary_tool_demo.csv"):
        self.backend = backend
        self.country = (country or "au").lower()
        self.csv_path = csv_path

        if self.backend != "csv":
            raise ValueError("Only CSV backend is supported in this setup.")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Salary CSV not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        # expected columns: role,state,annual_aud
        for col in ["role", "state", "annual_aud"]:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        df["_role_norm"] = df["role"].astype(str).str.strip().str.lower()
        df["_state_norm"] = df["state"].astype(str).str.strip().str.upper()
        df["_annual"] = pd.to_numeric(df["annual_aud"], errors="coerce")

        self.df = df
        self.known_roles = sorted(df["role"].astype(str).unique(), key=str.lower)

    def lookup(self, query: str, location: str = "Australia") -> Dict[str, Any]:
        """
        Compatible with app.py calling SAL.lookup(q).
        Parses role + state from free-text query.
        """
        role_guess, state_guess = _extract_role_and_state(query, self.known_roles)

        if not state_guess and location and location.lower() != "australia":
            state_guess = location.strip().upper()

        # filter by role
        rows = self.df[self.df["_role_norm"] == _norm(role_guess)]

        # filter by state if available
        if state_guess:
            rows = rows[rows["_state_norm"] == state_guess]

        if rows.empty:
            # try nationwide
            rows_any = self.df[self.df["_role_norm"] == _norm(role_guess)]
            if rows_any.empty:
                return {
                    "title": role_guess or "(unknown)",
                    "location": state_guess or "Australia",
                    "currency": "AUD",
                    "average": "N/A",
                    "samples": 0,
                    "backend": "csv",
                }
            avg = float(rows_any["_annual"].mean())
            return {
                "title": role_guess,
                "location": "Australia",
                "currency": "AUD",
                "average": round(avg, 2),
                "samples": int(len(rows_any)),
                "backend": "csv",
            }

        avg = float(rows["_annual"].mean())
        return {
            "title": role_guess,
            "location": state_guess or "Australia",
            "currency": "AUD",
            "average": round(avg, 2),
            "samples": int(len(rows)),
            "backend": "csv",
        }

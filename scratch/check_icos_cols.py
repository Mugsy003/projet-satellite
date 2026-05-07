from icoscp.dobj import Dobj
from config import PIDS_ICOS

sites_to_check = ["Lamasquere", "Voulundgaard"]

for site in sites_to_check:
    pid = PIDS_ICOS.get(site)
    print(f"\nSite: {site} (PID: {pid})")
    try:
        dobj = Dobj(pid)
        print("Columns:")
        # Print columns related to LW or Temperature
        relevant_cols = [c for c in dobj.colNames if 'LW' in c.upper() or 'TEMP' in c.upper() or 'TS' in c.upper()]
        print(relevant_cols)
        print("All columns:")
        print(dobj.colNames)
    except Exception as e:
        print(f"Error: {e}")

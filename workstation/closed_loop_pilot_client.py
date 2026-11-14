"""Reuse the native streaming client without changing active Table 1 code."""
import resident_table1_client as client
from closed_loop_pilot_worker import ResidentPool

if __name__ == '__main__':
    client.ResidentPool = ResidentPool
    client.main()

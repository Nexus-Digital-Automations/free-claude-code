# Tier implementations — each module exports a single apply() function.
# Counterpart: optimizer.py calls them in order: tier0 -> tier1 -> cache -> tier2.

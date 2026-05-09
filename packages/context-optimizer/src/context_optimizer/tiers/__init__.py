# Tier implementations — each module exports a single apply() function.
# Counterpart: optimizer.py calls them in order: tier0 -> tier0b/0c/0d/0e/0f -> tier1.
# Conversation-level compaction lives in block_tower/, not here.

import sys

path = "/root/verl-agent/agent_system/environments/env_package/extraction_ops/world.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = (
    '        lines = self._mission_lines() if self.procedural or include_brief else []\n'
    "        if include_brief:\n"
    '            lines.append("KNOWN MAP: " + "; ".join(f"{a}<->{b}" for a, b, *_ in self.edges))\n'
)

new = (
    "        lines = self._mission_lines() if self.procedural or self.random_maps or include_brief else []\n"
    "        if include_brief or self.random_maps:\n"
    '            lines.append("KNOWN MAP: " + "; ".join(f"{a}<->{b}" for a, b, *_ in self.edges))\n'
)

count = src.count(old)
if count != 1:
    print(f"ERROR: expected 1 match, found {count}")
    sys.exit(1)
src = src.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("world.py observation fix applied: random maps now show mission+map every step")

#!/usr/bin/env python3
"""
Repo Audit Script
Produces logs/repo_audit.json and logs/repo_audit.md
Includes file inventory, Python map, import graph, corruption heuristics, tactic discovery, etc.
Quarantines corrupted files to wip/recovery/ with .bak copies.
Logs moves in logs/recovery_moves.csv.
"""

import ast
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    wip_recovery = repo_root / 'wip' / 'recovery'
    wip_recovery.mkdir(parents=True, exist_ok=True)
    recovery_log = logs_dir / 'recovery_moves.csv'

    with open(recovery_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_path', 'new_path', 'reason'])

    audit = {}
    import_graph = defaultdict(list)
    tactics_found = set()
    entrypoints = []
    configs = []
    tests = []
    data_hooks = []
    ibkr_usage = []

    for root, dirs, files in os.walk(repo_root):
        for file in files:
            path = Path(root) / file
            rel_path = path.relative_to(repo_root)
            if str(rel_path).startswith(('logs/', 'wip/', '.git/')) or rel_path.name.startswith('.'):
                continue
            try:
                stat = path.stat()
                with open(path, 'rb') as f:
                    content = f.read()
                hash_ = hashlib.sha256(content).hexdigest()
                audit[str(rel_path)] = {
                    'size': stat.st_size,
                    'hash': hash_,
                    'mtime': stat.st_mtime,
                    'corrupted': False,
                    'reason': None
                }
                if path.suffix == '.py':
                    try:
                        tree = ast.parse(content.decode('utf-8', errors='ignore'))
                        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                        imports = []
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                imports.extend(alias.name for alias in node.names)
                            elif isinstance(node, ast.ImportFrom):
                                module = node.module or ''
                                imports.extend(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
                        audit[str(rel_path)]['python'] = {
                            'classes': classes,
                            'functions': functions,
                            'imports': imports,
                            'docstrings': [],  # Placeholder
                            'complexity': len(classes) + len(functions)
                        }
                        for imp in imports:
                            import_graph[str(rel_path)].append(imp)
                        # Entrypoints
                        if any(isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and
                               any(isinstance(comp, ast.Name) and comp.id == '__name__' for comp in ast.walk(node.test)) and
                               any(isinstance(comp, ast.Str) and comp.s == '__main__' for comp in ast.walk(node.test))
                               for node in ast.walk(tree)):
                            entrypoints.append(str(rel_path))
                        # IBKR
                        if 'ib_insync' in content.decode() or 'ib_async' in content.decode():
                            ibkr_usage.append(str(rel_path))
                        # Tactics
                        content_str = content.decode('utf-8', errors='ignore').lower()
                        if 'das' in content_str or 'tcea' in content_str or 'fvg' in content_str or 'vwap' in content_str or 'atr' in content_str:
                            tactics_found.add(str(rel_path))
                    except (SyntaxError, UnicodeDecodeError) as e:
                        audit[str(rel_path)]['corrupted'] = True
                        audit[str(rel_path)]['reason'] = str(e)
                        # Quarantine
                        bak_path = wip_recovery / f"{rel_path}.bak"
                        shutil.copy2(path, bak_path)
                        with open(recovery_log, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([str(rel_path), str(bak_path), str(e)])
                elif path.suffix in ('.yaml', '.yml', '.toml', '.env'):
                    configs.append(str(rel_path))
                elif 'test' in path.name or path.parent.name == 'tests':
                    tests.append(str(rel_path))
                elif 'databento' in content.decode('utf-8', errors='ignore').lower() or 'polygon' in content.decode('utf-8', errors='ignore').lower():
                    data_hooks.append(str(rel_path))
            except Exception as e:
                audit[str(rel_path)] = {
                    'size': 0,
                    'hash': '',
                    'mtime': 0,
                    'corrupted': True,
                    'reason': str(e)
                }

    # Output JSON
    with open(logs_dir / 'repo_audit.json', 'w') as f:
        json.dump({
            'audit': audit,
            'import_graph': dict(import_graph),
            'tactics_found': list(tactics_found),
            'entrypoints': entrypoints,
            'configs': configs,
            'tests': tests,
            'data_hooks': data_hooks,
            'ibkr_usage': ibkr_usage
        }, f, indent=2)

    # Output MD
    md_content = "# Repo Audit Report\n\n"
    md_content += f"## Summary\n- Total files: {len(audit)}\n"
    md_content += f"- Corrupted files: {sum(1 for v in audit.values() if v.get('corrupted'))}\n"
    md_content += f"- Python files: {sum(1 for k in audit if k.endswith('.py'))}\n"
    md_content += f"- Tactics found: {len(tactics_found)}\n"
    md_content += f"- Entrypoints: {len(entrypoints)}\n\n"

    md_content += "## File Inventory\n"
    for path, info in audit.items():
        md_content += f"- {path}: {info['size']} bytes, {'CORRUPTED' if info['corrupted'] else 'OK'}\n"
        if info.get('python'):
            md_content += f"  - Classes: {info['python']['classes']}\n"
            md_content += f"  - Functions: {info['python']['functions']}\n"

    with open(logs_dir / 'repo_audit.md', 'w') as f:
        f.write(md_content)

    print("Audit complete. Check logs/repo_audit.json and logs/repo_audit.md")

if __name__ == '__main__':
    main()

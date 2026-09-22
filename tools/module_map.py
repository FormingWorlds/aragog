#!/usr/bin/env python3
"""Print module, imports, and __all__ for key aragog packages."""

from __future__ import annotations

import ast
import os
from pathlib import Path

TARGET_PACKAGES = ['aragog', 'aragog/eos', 'aragog/solver', 'aragog/jax']
SRC_ROOT = Path(__file__).resolve().parent.parent / 'src'


def analyze_module(file_path: Path):
    rel_path = file_path.relative_to(SRC_ROOT)
    module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
    if module_name.endswith('.__init__'):
        module_name = module_name[:-9]

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=str(file_path))
        except Exception:
            return module_name, [], []

    imports = []
    all_exports = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            for alias in node.names:
                imports.append(f'{mod}.{alias.name}' if mod else alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                all_exports.append(elt.value)

    return module_name, sorted(set(imports)), sorted(all_exports)


def main():
    modules = []
    for pkg in TARGET_PACKAGES:
        pkg_dir = SRC_ROOT / pkg
        if not pkg_dir.exists():
            continue
        for root, _, files in os.walk(pkg_dir):
            for file in sorted(files):
                if file.endswith('.py'):
                    modules.append(Path(root) / file)

    lines = []
    for mod_path in sorted(modules):
        mod_name, imps, exports = analyze_module(mod_path)
        lines.append(f'Module: {mod_name}')
        lines.append('  Imports:')
        for imp in imps:
            lines.append(f'    {imp}')
        lines.append(f'  __all__: {exports}')

    output = '\n'.join(lines) + '\n'
    print(output, end='')

    out_file = SRC_ROOT.parent / 'tools' / 'verification' / 'module_map_baseline.txt'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(output)


if __name__ == '__main__':
    main()

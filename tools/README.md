# Project Tools

## 🔍 Hydra Target Resolver

This tool allows you to quickly open the Python file corresponding to a `_target_:` entry in any Hydra YAML config by pressing a custom keybinding in VS Code.  
It will jump to the actual class definition, even if the class is re-exported in an `__init__.py`.

---

### ✅ Setup Instructions

#### 1. Create a VS Code Task
In your `.vscode/tasks.json`, add:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Hydra: Go to _target_ at cursor",
      "type": "shell",
      "command": "python",
      "args": [
        "tools/resolve_target.py",
        "--at-cursor",
        "${file}",
        "${lineNumber}"
      ],
      "presentation": {
        "echo": true,
        "reveal": "silent",
        "focus": false,
        "panel": "shared"
      },
      "problemMatcher": []
    }
  ]
}
```

#### 2. Add a Keybinding
In `keybindings.json`, add the following keybinding for the key of your choice, e.g. `ctrl+f12`:

```json
{
  "key": "ctrl+f12",
  "command": "workbench.action.tasks.runTask",
  "args": "Hydra: Go to _target_ at cursor",
  "when": "editorTextFocus && editorLangId == yaml"
}
```

---

### 💡 Usage

1. Open any Hydra config file in VS Code.
2. Place your cursor on the line with the `_target_:` you want to resolve.
3. Press your keybinding (e.g. `Ctrl+F12`).
4. The script will open the corresponding Python file at the class definition, even if the class is re-exported.

---

**Tip:**  
If you want to resolve the first `_target_` in a file (not at cursor), you can run:
```sh
python tools/resolve_target.py --top-target <yaml_file>
```
Or resolve a specific target string:
```sh
python tools/resolve_target.py <python.module.ClassName>
```


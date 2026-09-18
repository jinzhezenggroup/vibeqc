import pathlib
src = pathlib.Path(r"D:\Users\Lenovo\Desktop\My_ViveQC\vibeqc\.claude\worktrees\issue-150-b-tiles\_runner_gpu.py").read_text()
src = src.replace(
    '    head = sh(["git", "-C", WORK, "rev-parse", "HEAD"]).strip()\n    log("WORKTREE_HEAD " + head)',
    '    head = pathlib.Path(f"{WORK}/.git/HEAD").read_text().strip() if pathlib.Path(f"{WORK}/.git/HEAD").exists() else "unknown"\n    if head.startswith("ref:"):\n        ref = head.split(None, 1)[1]\n        gitdir = pathlib.Path(WORK) / ".git"\n        gitdir = pathlib.Path(gitdir.read_text().split(None, 1)[1]) if gitdir.is_file() else gitdir\n        head = (gitdir / ref).read_text().strip()\n    log("WORKTREE_HEAD " + head)')
dst = pathlib.Path("/inspire/qb-ilm/project/chemicalreaction/czxs25220150/_150b/_runner_gpu.py")
dst.write_text(src, newline="\n")
print("PATCHED", len(src))
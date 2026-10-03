"""Reconstruct measured source without needing the unpublished local commit object."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True,help='Public checkout containing this publication and exact correction paths')
    parser.add_argument('--destination',type=Path,required=True,help='New local worktree; never overwritten')
    parser.add_argument('--baseline',action='store_true',help='Restore original public 1a4acc51 rejection source instead')
    args=parser.parse_args();repository=args.repository.resolve();destination=args.destination.resolve()
    provenance=json.loads((Path(__file__).with_name('provenance.json')).read_text())
    selected=provenance['original_source'] if args.baseline else provenance['measured_source']
    base=selected['revision'] if args.baseline else selected['public_base']
    def git(*argv):return subprocess.check_output(['git','-C',str(repository),*argv],text=True).strip()
    if destination.exists():raise FileExistsError(destination)
    # No network/fetch: a shallow clone must explicitly fetch the documented base.
    git('cat-file','-e',base+'^{commit}')
    files={}
    if not args.baseline:
        for name,row in selected['postimages'].items():
            raw=(repository/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Source postimage changed: '+name)
            files[name]=raw
    git('worktree','add','--detach',str(destination),base)
    for name,raw in files.items():
        path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    def target(*argv):return subprocess.check_output(['git','-C',str(destination),*argv],text=True).strip()
    if files:target('add','--',*files)
    tree=target('write-tree')
    if tree!=selected['tree']:raise ValueError('Reconstructed tree mismatch; preserve destination for diagnosis')
    if files:
        target('-c','user.name=Scientific reproduction','-c','user.email=reproduction@localhost','commit','-m','Reconstruct measured CPU KS closure source from exact public postimages')
    final_tree=target('rev-parse','HEAD^{tree}')
    if final_tree!=selected['tree'] or target('status','--porcelain'):
        raise ValueError('Reconstruction commit or hooks changed the expected clean tree; preserve destination for diagnosis')
    print(json.dumps(dict(destination=str(destination),head=target('rev-parse','HEAD'),tree=final_tree,measured_revision=selected['revision'],same_tree=True)))


if __name__=='__main__':main()

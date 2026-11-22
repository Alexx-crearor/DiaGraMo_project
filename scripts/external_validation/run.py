"""Stage-gated research runner; explicit local imports, no historical report imports."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
MAX_RUNTIME_HOURS=8


def load(name):
    if name not in {'common','external_reader','stages','reports','experiments'}:raise ValueError('Unapproved module')
    spec=importlib.util.spec_from_file_location('diagramo_external_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert Path(m.__file__).resolve()==HERE/(name+'.py')
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['all','0','1','continue','validate'],default='all')
    ap.add_argument('--max-runtime-hours',type=float,default=MAX_RUNTIME_HOURS)
    ap.add_argument('--report-reserve-minutes',type=float,default=15)
    args=ap.parse_args();c=load('common');budget=c.Budget(args.max_runtime_hours,args.report_reserve_minutes)
    if args.phase in {'all','0'}:c.initialize(budget);print('Phase0 persisted',flush=True)
    if args.phase in {'all','1'}:load('stages').acquire(c,load('external_reader'),budget);print('Phase1 persisted',flush=True)
    if args.phase in {'all','continue'}:load('stages').continue_research(c,load('external_reader'),budget,load)
    if args.phase=='validate':load('stages').validate(c,budget)


if __name__=='__main__':main()

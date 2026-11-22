"""Bounded source downloads and safe ZIP/XLSX inspection, never execute dataset code."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import time
import zipfile
import xml.etree.ElementTree as ET


def inspect_zip(path,max_uncompressed=512*1024*1024,max_files=10000):
    with zipfile.ZipFile(path) as z:
        infos=z.infolist()
        if len(infos)>max_files or sum(i.file_size for i in infos)>max_uncompressed:raise ValueError('Archive limit exceeded')
        names=set()
        for i in infos:
            name=i.filename.replace('\\','/')
            if '\x00' in name or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive or '..' in PurePosixPath(name).parts or ':' in name:raise ValueError('Unsafe ZIP path')
            if name.casefold() in names:raise ValueError('Duplicate/case-colliding archive path')
            names.add(name.casefold())
            mode=i.external_attr>>16
            if stat.S_ISLNK(mode):raise ValueError('Archive symlink rejected')
            if i.flag_bits&1:raise ValueError('Encrypted archive rejected')
            if Path(name).suffix.lower() in {'.exe','.dll','.bat','.cmd','.ps1','.com','.scr','.msi','.pkl','.pickle','.joblib'}:raise ValueError('Executable/unsafe serialization rejected')
            if i.file_size>max_uncompressed or (i.compress_size and i.file_size/i.compress_size>1000):raise ValueError('Suspicious expansion ratio')
        bad=z.testzip()
        if bad:raise ValueError('Archive CRC failure')
        return [dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,directory=i.is_dir()) for i in infos]


def extract_safe(path,directory):
    inventory=inspect_zip(path);root=Path(directory).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for row in inventory:
            destination=(root/row['name'].replace('\\','/')).resolve()
            if not destination.is_relative_to(root):raise ValueError('Extraction escape')
            if row['directory']:destination.mkdir(parents=True,exist_ok=True);continue
            data=z.read(row['name'])
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if destination.read_bytes()!=data:raise ValueError('Existing extracted file differs; refusing overwrite')
            else:
                with destination.open('xb') as f:f.write(data)
    return inventory


def download(url,dest,max_bytes,expected_git_blob=None,expected_bytes=None):
    import requests
    if not url.startswith(('https://raw.githubusercontent.com/peet292929/Dysgraphia-detection-through-machine-learning/',
                           'https://api.github.com/repos/peet292929/Dysgraphia-detection-through-machine-learning/')):
        raise ValueError('URL outside explicit research allowlist')
    dest=Path(dest)
    if not dest.exists():
        temp=dest.with_suffix(dest.suffix+'.part');dest.parent.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        try:
            with requests.get(url,stream=True,timeout=(10,30),allow_redirects=False) as r:
                r.raise_for_status()
                if r.is_redirect:raise ValueError('Unexpected download redirect')
                if int(r.headers.get('Content-Length','0'))>max_bytes:raise ValueError('Download too large')
                total=0
                with temp.open('wb') as f:
                    for chunk in r.iter_content(256*1024):
                        total+=len(chunk)
                        if total>max_bytes or time.monotonic()-started>180:raise ValueError('Bounded download exceeded')
                        f.write(chunk)
            os.replace(temp,dest)
        except Exception:
            # Preserve incomplete .part for diagnosis, never treat it as completed source.
            raise
    raw=dest.read_bytes()
    if len(raw)>max_bytes or (expected_bytes is not None and len(raw)!=expected_bytes):raise ValueError('Source size mismatch')
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if expected_git_blob and blob!=expected_git_blob:raise ValueError('Git blob mismatch')
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=blob,url=url)


def xlsx_cells(path):
    """Read cell values/styles using OOXML, without interpreting formulas or macros."""
    inspect_zip(path,64*1024*1024,2000)
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    relns='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    with zipfile.ZipFile(path) as z:
        if any(n.lower().endswith('vbaproject.bin') for n in z.namelist()):raise ValueError('Macros rejected')
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings=[''.join(t.text or '' for t in si.findall('.//s:t',ns)) for si in root.findall('s:si',ns)]
        relationships=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets={r.attrib['Id']:r.attrib['Target'] for r in relationships}
        workbook=ET.fromstring(z.read('xl/workbook.xml'));result={}
        for sheet in workbook.findall('s:sheets/s:sheet',ns):
            t=targets[sheet.attrib[relns]]
            sheetpath=t.lstrip('/') if t.startswith('/') else 'xl/'+t
            root=ET.fromstring(z.read(sheetpath));cells=[]
            for c in root.findall('.//s:sheetData/s:row/s:c',ns):
                val=c.find('s:v',ns);value=val.text if val is not None else ''
                if c.attrib.get('t')=='s':value=strings[int(value)]
                elif c.attrib.get('t')=='inlineStr':value=''.join(x.text or '' for x in c.findall('.//s:t',ns))
                cells.append(dict(cell=c.attrib['r'],value=value or '',type=c.attrib.get('t','numeric'),
                    formula_present=c.find('s:f',ns) is not None,
                    formula=c.find('s:f',ns).text if c.find('s:f',ns) is not None else None,style=c.attrib.get('s')))
            result[sheet.attrib['name']]=cells
        return result


def require_raw_contract(channels,units_verified,task_mapping_verified):
    required={'x','y','time','pen_status','pressure','azimuth','tilt'}
    if set(channels)!=required:raise ValueError('Missing/unverified channels; no fabrication allowed')
    if not units_verified or not task_mapping_verified:raise ValueError('Semantic contract not verified')
    return True


def load_svc_numeric(path):
    import numpy as np
    with Path(path).open('r',encoding='ascii') as f:
        header=f.readline().strip()
        if not header.isdecimal():raise ValueError('Invalid SVC sample count header')
        if int(header)==0:raise ValueError('Empty SVC: zero declared samples; not a valid signal recording')
        values=np.loadtxt(f,dtype=np.float64,ndmin=2)
    if values.shape[1]!=7 or len(values)!=int(header):raise ValueError('SVC header/column mismatch')
    if not np.isfinite(values).all():raise ValueError('SVC nonfinite values')
    return values


def workbook_participants(sheets):
    """Only explicit workbook ID+diag rows; cached simple references checked, not executed."""
    import pandas as pd
    result=[]
    for sheet,cells in sheets.items():
        bycell={x['cell']:x for x in cells}
        rows={}
        for x in cells:
            m=re.fullmatch(r'([A-Z]+)(\d+)',x['cell'])
            rows.setdefault(int(m[2]),{})[m[1]]=x['value'].strip()
        if [rows.get(1,{}).get(x) for x in 'ABCDE']!=['ID','diag','sex','hand','age']:raise ValueError('Unknown workbook schema')
        for n,row in sorted(rows.items()):
            if n==1 or not row.get('A'):continue
            source_id=row['A']
            if not re.fullmatch(r'\d{1,5}',source_id):raise ValueError('Invalid source participant ID')
            # Excel mixes textual zero-padded IDs and numeric IDs; SVC paths use five digits.
            # Padding all numeric IDs is label-independent and collisions are rejected below.
            pid=source_id.zfill(5)
            label=row.get('B')
            if label not in {'DYSGR','0'}:raise ValueError('Unsupported/missing clinical label')
            cell=bycell['B'+str(n)]
            if cell.get('formula_present'):
                formula=cell.get('formula') or ''
                reference=formula.lstrip('+=')
                if not re.fullmatch(r'B\d+',reference) or reference not in bycell or bycell[reference]['value'].strip()!=label:
                    raise ValueError('Unverified cached label formula')
            result.append(dict(participant_id=pid,workbook_id_original=source_id,id_padding_applied=len(source_id)<5,label=int(label=='DYSGR'),source_label=label,
                sex=row.get('C','').strip(),hand=row.get('D','').strip(),age=row.get('E',''),sheet=sheet,
                label_formula_present=cell.get('formula_present',False)))
    frame=pd.DataFrame(result)
    if not len(frame) or not frame.participant_id.is_unique:raise ValueError('Ambiguous workbook IDs')
    return frame

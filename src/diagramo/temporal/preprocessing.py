"""Timestamp-safe, sample-preserving channels in original acquisition order."""
import numpy as np
RAW_CHANNELS=('x','y','time','pen_status','pressure','azimuth','tilt')
CHANNELS=('relative_x','relative_y','pressure','azimuth','tilt','elapsed_time','delta_time','pen_down',
          'velocity_x','velocity_y','speed','acceleration_x','acceleration_y','velocity_valid','acceleration_valid',
          'dt_valid','duplicate_time','negative_time','tiny_interval','large_gap','boundary')
CONTINUOUS=tuple(range(7))+tuple(range(8,13))
TIME_CHANNELS=(5,6)

def preprocess(record,minimum_dt=1e-4,maximum_dt=1.):
    raw={c:np.asarray(record[c],np.float64)for c in RAW_CHANNELS}
    n=len(raw['time']);assert n>=1 and all(len(a)==n and np.isfinite(a).all()for a in raw.values())
    t=raw['time'];pen=raw['pen_status'];assert set(np.unique(pen))<= {0,1}
    dt=np.r_[0.,np.diff(t)];valid=(dt>=minimum_dt)&(dt<=maximum_dt);valid[0]=False
    pen_change=np.r_[True,pen[1:]!=pen[:-1]]
    boundary=pen_change|~valid;boundary[0]=True
    segment=np.cumsum(boundary)-1;run=np.cumsum(pen_change)-1
    velocity_valid=valid&~pen_change&(pen==1)
    vx=np.zeros(n);vy=np.zeros(n)
    np.divide(np.r_[0.,np.diff(raw['x'])],dt,out=vx,where=velocity_valid)
    np.divide(np.r_[0.,np.diff(raw['y'])],dt,out=vy,where=velocity_valid)
    acceleration_valid=velocity_valid&np.r_[False,velocity_valid[:-1]]
    ax=np.zeros(n);ay=np.zeros(n)
    np.divide(np.r_[0.,np.diff(vx)],dt,out=ax,where=acceleration_valid)
    np.divide(np.r_[0.,np.diff(vy)],dt,out=ay,where=acceleration_valid)
    values=np.column_stack([raw['x']-raw['x'][0],raw['y']-raw['y'][0],raw['pressure'],raw['azimuth'],raw['tilt'],
       t-t[0],np.where(valid,dt,0.),pen,vx,vy,np.hypot(vx,vy),ax,ay,velocity_valid,acceleration_valid,valid,
       dt==0,dt<0,(dt>0)&(dt<minimum_dt),dt>maximum_dt,boundary])
    values[0,16:20]=0. # first sample has no previous interval, not a duplicate
    physical=values.copy()
    values[:,CONTINUOUS]=np.sign(values[:,CONTINUOUS])*np.log1p(np.abs(values[:,CONTINUOUS]))
    assert np.isfinite(values).all()
    # An index-only control recalculates motion from sample differences, never seconds.
    index_values=physical.copy();index_values[:,5:7]=0.
    iv=(pen==1)&~pen_change;dx=np.r_[0.,np.diff(raw['x'])];dy=np.r_[0.,np.diff(raw['y'])]
    dx[~iv]=0.;dy[~iv]=0.;ia=iv&np.r_[False,iv[:-1]]
    ddx=np.r_[0.,np.diff(dx)];ddy=np.r_[0.,np.diff(dy)];ddx[~ia]=0.;ddy[~ia]=0.
    index_values[:,8:15]=np.column_stack([dx,dy,np.hypot(dx,dy),ddx,ddy,iv,ia])
    index_values[:,15:20]=0.
    index_values[:,CONTINUOUS]=np.sign(index_values[:,CONTINUOUS])*np.log1p(np.abs(index_values[:,CONTINUOUS]))
    return dict(values=values.astype('float32'),index_values=index_values.astype('float32'),physical=physical,
        time=t,pen=pen.astype('int8'),segment=segment.astype('int32'),run=run.astype('int32'),boundary=boundary,
        raw_index=np.arange(n,dtype='int32'),dt=dt,velocity_valid=velocity_valid,acceleration_valid=acceleration_valid)


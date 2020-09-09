import xarray as xr

ifile = sys.argv[1] 
ofile = sys.argv[2] 
lon   = sys.argv[3] 
lat   = sys.argv[4] 
time  = sys.argv[5] 
def export_NetCDF(ifile,ofile,lon,lat,time):

    # Reload saved GE-NN results 
    with open(ifile, 'rb') as handle: 
        FP_GENN = pickle.load(handle)[3]

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    xrdata = xr.Dataset(\
                data_vars={'longitude': (('lat','lon'),mesh_lon),\
                           'latitude' : (('lat','lon'),mesh_lat),\
                           'Time'     : (('time'),time),\
                           'FP-GENN'  : (('time','lat','lon'),FP_GENN)},\
                coords={'lon': lon,'lat': lat,'time': range(0,len(time))})
    xrdata.time.attrs['units']='days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=ofile, mode='w')

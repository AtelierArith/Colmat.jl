import sys,tomllib,statistics,json,gzip
def read(path):
    with (gzip.open(path,"rb") if path.endswith(".gz") else open(path,"rb")) as stream:
        return tomllib.load(stream)
before,after=(read(p) for p in sys.argv[1:3])
for field in ('native_revision','tool_manifest_sha256','julia_version','cpu_model','platform','validation_blas'):
    assert before[field]==after[field], field

keys=('scalar','n','left','right','balance','participants','fixture')
def key(row):return tuple(row[k] for k in keys)
b={key(r):r for r in before['cases']};a={key(r):r for r in after['cases']}
assert a.keys()==b.keys() and len(a)==768
assert all(r['info']==0 and r['samples']==10 and r['julia_allocations']==0 and r['julia_memory_bytes']==0 for r in a.values())
result={'before_revision':before['source_revision'],'after_revision':after['source_revision'],'cases':len(a),'groups':[]}
for n in (32,96,256):
 for mode in ('NN','VN','NV','VV'):
  ks=[k for k in a if a[k]['n']==n and a[k]['left']+a[k]['right']==mode]
  row={'n':n,'mode':mode,'median_before_over_after':statistics.median(b[k]['steady_median_ns']/a[k]['steady_median_ns'] for k in ks)}
  if 'native_profile' in next(iter(a.values())):
   for metric in ('median_allocations','median_allocated_bytes','median_extra_peak_live_bytes'):
    row[metric+'_before']=statistics.median(b[k]['native_profile'][metric] for k in ks)
    row[metric+'_after']=statistics.median(a[k]['native_profile'][metric] for k in ks)
  result['groups'].append(row)
result['max_residuals']={s:max(r.get(side+'_residual',0) for r in a.values() if r['scalar']==s for side in ('left','right')) for s in ('Float32','Float64','ComplexF32','ComplexF64')}
print(json.dumps(result,indent=2))

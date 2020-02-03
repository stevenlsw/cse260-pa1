import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

wb = xlrd.open_workbook("results.xlsx")
sheet = wb.sheet_by_index(0)

N = []
results = []
for i in range(1, sheet.nrows):
    N.append(sheet.cell_value(i, 0))
    for j in range(1, sheet.ncols):
        results.append(sheet.cell_value(i,j))
results = np.array(results,dtype=np.float32).reshape(sheet.nrows-1, sheet.ncols-1)

methods = ['naive','1 level block','3 level block', 'avx 4*4 w/ buffer',
           'avx 4*4 w/o buffer', 'avx 3*12 w/ L1 buffer',
           'avx 3*12 w/ L3 buffer', 'best optimized', 'blas']
markers = ['.','o','^','8','s','p','X','D','*']

plt.figure()
values = range(len(methods))
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

N_display = np.arange(len(N),dtype=np.int)+1
for i in range(len(methods)):
    method = methods[i]
    color = scalarMap.to_rgba(i)
    marker = markers[i]
    plt.plot(N_display, results[:,i], label=method, color=color, marker=marker)

plt.xlabel('20 different matrix sizes')
plt.ylabel('GFlops')
plt.legend(loc='best')
plt.xticks(N_display)
plt.savefig('performance.png')
plt.show()



xmatch
======

xmatch creates a cross matched catalog of targets based on the Ecliptic Input Catalog. Matches are based on both distance and magnitude. Cuts are applied in probability to choose which matches are 'hard' matches, 'soft' matches or missing sources. The [tutorial](https://github.com/christinahedges/xmatch/blob/master/crossmatching_tutorial.ipynb) describes how the cross matching works. The cross match can be run using

'''python
import xmatch
xmatch.match('input.fits',depth=5)
xmatch.fit()
'''

Where *input.fits* is a fits file containing RAs, Decs and magnitudes for each target. 

Starmap
-------

xmatch contains a simple function for plotting starmaps with nearby sources. The call 

'''python
import matplotlib.pyplot as plt
import pandas as pd
import xmatch
results=pd.read_pickle('results.p')
fig,ax=fig.subplots(1,figsize=(7,6))
xmatch.starmap(0,results,ax,cbar=True)
'''


<table border="0">
<tr>
<td><img src=images/blended_example.png style="width: 500px;"></td>
<td><img src=images/crowded_example.png style="width: 500px;"></td>
</tr>
</table>
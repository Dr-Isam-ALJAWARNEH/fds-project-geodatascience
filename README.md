# Geospatial Data Science (GeoDataScience)


The task is to extend Datascience Table abstraction so that it supports geospatial data science. 
As a starting point, it reads csv containing geospatial data in form of longitude latitude and then use geodataframe 
to create new table (you can call the table geotable ) in this case mimicking the table abstraction and extending it to include 
processing of geospatial data, of a special interest is converting pairs of longitude latitude into geometry just like in geodataframe!

# Guidelines

- I am providing herein some guidelines available in [Guidelines Markdown file](Guidelines.md) of what the team need to implement (mostly) for the geodatascience project. However, this is just a guideline and you can (and should ) expand as per convenience and agreement between team members and me.

To learn more about geospatial data analysis generally, and to learn what other features you can support, refer to the following online resources:
1. [Geospatial Analysis with Python](https://kodu.ut.ee/~kmoch/geopython2020/index.html)
2. [Geospatial Data Science in Python](https://zia207.github.io/geospatial-python.io/index.html)
3. [Geocomputation with Python](https://py.geocompx.org/)
4. [Spatial Data Science](https://r-spatial.org/book/)
5. [Geospatial-Data-Science-Quick-Start-Guide](https://github.com/PacktPublishing/Geospatial-Data-Science-Quick-Start-Guide)
6. [Geographic Data Science with PySAL and the pydata stack](https://darribas.org/gds_scipy16/)

> You can add any feature you deem appropriate to the fr`geodatascience` abstraction. However, the most important part is to add support for every feature that is available in the `datascience` abstraction.
have a look at some of the features in the `datascience` available [here](https://www.data8.org/datascience/) and [here](https://github.com/data-8/datascience). However, the idea is to extend the `datascience` abstraction so that the new `geodatascience` abstraction support features similar to those in the `datascience` and adds support to many spatial operations and featrures, specially the baseline features and spatial operations, such as spatial join.




We will extend the following abstraction:
- [datascience](https://github.com/data-8/datascience).

***Team members***:
- [Dr. Isam Al Jawarneh](https://isamaljawarneh.github.io/) (```supervisor```)
- [Amal Ali Ali] (```member```)
- [Joude Azzam] (```member```)
- [Sara Alketbi] (```member```)
- [Siham Alkousa] (```member```)




- To extend the abstraction of converting a CSV into a table to handle geospatial data, we can create a new class GeoTable that inherits from the existing Table class. This new class will include functionality to process geospatial data, such as converting pairs of longitude and latitude into geometry objects, similar to how GeoDataFrames work in the geopandas library.

Example Usage
Input CSV File (`locations.csv`):

```
name,longitude,latitude
LocationA,-122.4194,37.7749
LocationB,-118.2437,34.0522
LocationC,-73.9352,40.7306
```

Output
1. GeoTable :
```
name      | longitude | latitude | geometry
LocationA | -122.4194 | 37.7749  | POINT (-122.4194 37.7749)
LocationB | -118.2437 | 34.0522  | POINT (-118.2437 34.0522)
LocationC | -73.9352  | 40.7306  | POINT (-73.9352 40.7306)
```

2. Plot :
- A map showing the locations as points.

This implementation provides a seamless way to handle geospatial data within the Table abstraction while leveraging the power of `geopandas`.

 - Also, this abstraction should be able to convert longitude latitude pairs to geohash
    > To extend the GeoTable abstraction to include the conversion of longitude and latitude pairs into geohashes, we can integrate the geohash library. Geohash is a system that encodes geographic coordinates (latitude and longitude) into a short string of letters and digits. This encoding is useful for proximity searches and spatial indexing.



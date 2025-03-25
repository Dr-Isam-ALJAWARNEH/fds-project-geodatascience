


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
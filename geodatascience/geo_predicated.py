from datascience.predicates import are as base_are, _combinable, check_iterable
from shapely.geometry import Point, Polygon, LineString  # Import specific geometry types
import warnings

class are(base_are):
    """
    Extends the datascience are class with geospatial predicates.
    Now uses proper Shapely imports for version compatibility.
    """

    @staticmethod
    def within(other_geometry):
        """
        Creates a predicate checking if geometries are within another shape.
        
        Args:
            other_geometry: Shapely Polygon/MultiPolygon
            
        Example:
            >>> from shapely.geometry import Polygon
            >>> area = Polygon([[0,0], [1,0], [1,1], [0,1]])
            >>> table.where("geometry", are.within(area))
        """
        check_iterable(other_geometry)
        if not hasattr(other_geometry, 'within'):  # More flexible check
            raise TypeError(
                f"Expected Shapely geometry, got {type(other_geometry)}. "
                "Convert with: `from shapely.geometry import shape`"
            )
        return _combinable(lambda g: g.within(other_geometry))



    @staticmethod
    def intersects(other_geometry):
        """
        Creates a predicate checking for geometric intersections.
        
        Args:
            other_geometry: Any Shapely geometry (Point, LineString, etc.)
            
        Example:
            >>> roads.where("geometry", are.intersects(city_boundary))
        """
        check_iterable(other_geometry)
        if not hasattr(other_geometry, 'intersects'):
            warnings.warn(
                f"Argument {other_geometry} is not a Shapely geometry. "
                "Results may be unexpected.",
                UserWarning
            )
        return _combinable(lambda g: g.intersects(other_geometry))
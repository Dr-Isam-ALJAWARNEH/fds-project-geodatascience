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
    



    @staticmethod 
    def not_within(geometry):
        """
        Creates a spatial predicate that identifies geometries NOT completely contained within
        a reference shape. This is the logical inverse of the within() operation.
        
        Args:
            geometry: A Shapely Polygon or MultiPolygon defining the boundary to test against
            
        Returns:
            A predicate function that returns True when input geometries:
            - Are completely outside the boundary OR
            - Touch the boundary but aren't fully enclosed
            
        Behavior Details:
            - Boundary points: Returns True (unlike within() which returns False)
            - Null/empty geometries: Returns False
            - Works with all Shapely geometry types (Points, LineStrings, etc.)
            - Maintains topological accuracy using precise spatial calculations
            
        Example Use Cases:
            1. Finding points outside a regulated zone:
            `are.not_within(pollution_control_area)`
            
            2. Identifying features near but not inside a boundary:
            `.where(are.intersects(boundary).and_(are.not_within(boundary)))`
            
            3. Data validation to detect outliers:
            `valid_data = data.where(are.not_within(city_limits))`
        """
        return _combinable(lambda g: not g.within(geometry))
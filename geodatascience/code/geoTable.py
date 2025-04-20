from datascience import Table

class GeoTable(Table):
    def __init__(self, *args, **kwargs):
        # Call the base Table constructor
        super().__init__(*args, **kwargs)

        # Ensure 'geometry' column exists
        if 'geometry' not in self.labels:
            self.append_column('geometry', [None] * self.num_rows)

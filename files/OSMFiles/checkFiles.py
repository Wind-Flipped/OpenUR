import fiona
import shapefile

path = './shape/landuse.shp'
# 不是 points path = './Beijing-shp/shape/points.shp'
# 不是 places path = './Beijing-shp/shape/places.shp'
# 不是，但是这里面有一些购物广场之类的 path = './Beijing-shp/shape/landuse.shp'
# 不是，path = './Beijing-shp/shape/buildings.shp'
# 不是其他的 path = './Beijing-shp/shape/waterways.shp'
file = shapefile.Reader(path)

# print each boundary in this file

for sr in file.shapeRecords():
    # print(sr.shape.bbox)
    if sr.record[1] != "":
        print(sr.record)
        print(sr.shape.points)

    # print sr's name

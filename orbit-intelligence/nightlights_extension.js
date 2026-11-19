// OPTIONAL, untested here: paste into Google Earth Engine Code Editor (requires account).
// Extract port-area radiance from NOAA VIIRS MONTHLY_V1_VCMCFG. Does not use NASA Black Marble.
// Port-area rectangle near western Singapore terminals; exploratory boundary, not certified port polygon.
var roi = ee.Geometry.Rectangle([103.60,1.22,103.77,1.35]);
var images = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
  .filterDate('2019-01-01','2025-04-01').filterBounds(roi);
var results = ee.FeatureCollection(images.map(function(im){
  var valid=im.select('cf_cvg').gte(3);
  var stats=im.select('avg_rad').updateMask(valid).reduceRegion({reducer:ee.Reducer.mean(),geometry:roi,scale:500,maxPixels:1e8});
  var coverage=valid.reduceRegion({reducer:ee.Reducer.mean(),geometry:roi,scale:500,maxPixels:1e8});
  return ee.Feature(null,{month:im.date().format('YYYY-MM-01'),radiance:stats.get('avg_rad'),valid_fraction:coverage.get('cf_cvg')});
}));
print(results.limit(5)); Map.centerObject(roi,11); Map.addLayer(roi,{color:'red'},'Exploratory port area');
Export.table.toDrive({collection:results,description:'Singapore_VIIRS_monthly',fileFormat:'CSV'});
// After exporting, merge to singapore_monthly.csv by month; test with chronological split and a baseline.
// Avoid interpreting illumination alone as economic activity; cloud coverage, new infrastructure and spill light confound.

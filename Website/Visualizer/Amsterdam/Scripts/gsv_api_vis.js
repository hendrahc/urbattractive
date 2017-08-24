var map;
var marker_sel;
var markers;
var sym_sel;
var color_code = {0: "darkgrey", 1: "red", 2: "orange", 3:"yellow", 4:"cyan", 5:"blue"};
var heatmap;
var heatMapData;

function initMap_b() {
	var myLatLng = {lat: 52.35, lng: 4.85};

	map = new google.maps.Map(document.getElementById('map'), {
	  zoom: 12,
	  center: myLatLng
	});

	var symbol = [];
	
	for (var v=0; v<=5; v++){
		symbol[v] = {
			  path: google.maps.SymbolPath.CIRCLE,
			  scale: 3,
			  strokeWeight: 3,
			  strokeColor: color_code[v],
			  fillColor: color_code[v],
			  fillOpacity: 0.8,
			};
	}
	
	
	sym_sel = {
      path: google.maps.SymbolPath.CIRCLE,
      scale: 8,
	  fillColor: "darkgreen",
	  strokeWeight: 2,
	  strokeColor: 'black',
	  fillOpacity: 0.8,
    };
	
	markers = {}

	marker_sel = new google.maps.Marker({
				map: map,
				icon: sym_sel
			});
	
	for(var rec in coordinates_data){
		var pos = {};
		pos["lat"] = parseFloat(coordinates_data[rec]["lat"]);
		pos["lng"] = parseFloat(coordinates_data[rec]["long"]);
		
		var name = coordinates_data[rec]["name"].toString();
		var val = coordinates_data[rec]["overall_attractiveness"];
		
		markers[name] = new google.maps.Marker({
			position: pos,
			map: map,
			icon: symbol[val],
			title: name
		});
		
		markers[name].addListener('click', function() {
			selectLoc(this.getTitle());
        });
		
	}
	selectLoc(name);
}

function generate_heatmapdata(){
	var h = [];
	for(var rec in coordinates_data){
		var lc = {}
		lc["lat"] = parseFloat(coordinates_data[rec]["lat"]);
		lc["long"] = parseFloat(coordinates_data[rec]["long"]);
		lc["weight"] = coordinates_data[rec]["overall_attractiveness"];
		h.push(lc);
	}
	
	return {max:5, data: h};
}

function initMap() {
	var myLatLng = {lat: 52.35, lng: 4.85};

	heatMapData = generate_heatmapdata();
	
	
	map = new google.maps.Map(document.getElementById('map'), {
	  zoom: 12,
	  center: myLatLng
	});

	heatmap = new HeatmapOverlay(map, 
		{
			// radius should be small ONLY if scaleRadius is true (or small radius is intended)
			//"radius": 0.006,
			"radius": 0.015,
			"maxOpacity": 1, 
			// scales the radius based on map zoom
			"scaleRadius": true, 
			// if set to false the heatmap uses the global maximum for colorization
			// if activated: uses the data maximum within the current map boundaries 
			//   (there will always be a red spot with useLocalExtremas true)
			"useLocalExtrema": true,
			latField: 'lat',
			lngField: 'long',
			valueField: 'weight'
		}
	);
	
	heatmap.setData(heatMapData);
	
}

function selectLoc(loc_id){
	marker_sel.setPosition(markers[loc_id].getPosition());
	var v = coordinates_data[loc_id]["overall_attractiveness"];
	sym_sel.fillColor = color_code[v];
	marker_sel.setIcon(sym_sel);
	showIm(loc_id);
}
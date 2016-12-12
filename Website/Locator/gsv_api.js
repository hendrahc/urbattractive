function initMap() {
	var myLatLng = {lat: 52.35, lng: 4.85};

	var map = new google.maps.Map(document.getElementById('map'), {
	  zoom: 12,
	  center: myLatLng
	});

	var symbol = {
      path: google.maps.SymbolPath.CIRCLE,
      scale: 3,
	  strokeWeight: 3,
	  strokeColor: 'red',
	  fillOpacity: 0.8,
    };
	
	var markers = {}

	for(var rec in coordinates_data){
		var pos = {};
		pos["lat"] = parseFloat(coordinates_data[rec]["lat"]);
		pos["lng"] = parseFloat(coordinates_data[rec]["long"]);
		
		var name = coordinates_data[rec]["name"];
		
		markers[name] = new google.maps.Marker({
			position: pos,
			map: map,
			icon: symbol,
			title: name
		});
	}

}
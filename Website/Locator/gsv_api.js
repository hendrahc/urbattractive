var imgLoc = "../../../Dataset/PILOT/";

function showIm(loc_id) {
	var i1 = document.getElementById('img1');
	i1.src = imgLoc+"GSV_PILOT_"+loc_id+"_1.jpg";
	
	var i2 = document.getElementById('img2');
	i2.src = imgLoc+"GSV_PILOT_"+loc_id+"_2.jpg";
	
	var i3 = document.getElementById('img3');
	i3.src = imgLoc+"GSV_PILOT_"+loc_id+"_3.jpg";
	
	var i4 = document.getElementById('img4');
	i4.src = imgLoc+"GSV_PILOT_"+loc_id+"_4.jpg";
}
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
	
	var sym_sel = {
      path: google.maps.SymbolPath.CIRCLE,
      scale: 4,
	  strokeWeight: 3,
	  strokeColor: 'green',
	  fillOpacity: 0.9,
    };
	
	var markers = {}

	var marker_sel = new google.maps.Marker({
				map: map,
				icon: sym_sel
			});
	
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
		
		markers[name].addListener('click', function() {
			marker_sel.setPosition(this.getPosition());
			showIm(this.getTitle());
			
        });
		
	}
	

}
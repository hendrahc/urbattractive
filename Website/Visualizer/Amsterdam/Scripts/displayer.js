
$(document).ready(function(){
        $("#imgshow").ready(function(){
            for (var n = 1; n <= 4; ++ n){
                var cont = $('<div />', {class: "col-md-2 cont_img", id:"cont_img"+n})
                cont.appendTo($("#imgshow"));

                var ni = $('<img />',{class:"img-thumbnail img_thumb", id:"img"+n, alt:"img"+n});
                ni.appendTo(cont);

                var desc = $('<div />', {class: "desc_img", id:"desc_img"+n})
                desc.text("Test desc")
                desc.appendTo(cont);
            }
        });

        show_gallery();
    });


var imgLoc = "../crowdsourcing/public/images/";

function showIm(loc_id) {
    for (var n = 1; n <= 4; ++ n){
        var ii = document.getElementById('img'+n);
        ii.src = imgLoc+coordinates_data[loc_id]["imgs"]["img"+n]["filepath"];
		var desc_i = $('#desc_img'+n);
		desc_i.text("attr= "+coordinates_data[loc_id]["imgs"]["img"+n]["attractiveness"]);
    }
}

function show_gallery() {
    for(var rec in coordinates_data){
		var loc_cont = $('<div/>',{class:"loc_cont"});
		loc_cont.appendTo($("#gallery-content"))
        var loc_id = coordinates_data[rec]["name"];
        for (var n = 1; n <= 4; ++ n){
            var src = imgLoc+coordinates_data[rec]["imgs"]["img"+n]["filepath"];
            var img =  $('<img />',{class:"img-thumbnail img_thumb_gal", id:"img_"+loc_id+"_"+n, alt:loc_id+"_img"+n, src:src});
            img.appendTo(loc_cont);
			img.click(function(){
				val = this.alt.split("_")[0];
				selectLoc(val);
			});
        }
		var locator = $('<div/>',{class:"img-thumbnail"});
		locator.text(loc_id);
		locator.css("background",color_code[coordinates_data[loc_id]["overall_attractiveness"]])
		locator.appendTo(loc_cont);
    }
}
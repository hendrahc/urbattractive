
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


var imgLoc = "../../../Dataset/PILOT/";

function showIm(loc_id) {
    for (var n = 1; n <= 4; ++ n){
        var ii = document.getElementById('img'+n);
        ii.src = imgLoc+"GSV_PILOT_"+loc_id+"_"+n+".jpg";
    }
}

function show_gallery() {
    for(var rec in coordinates_data){
        var loc_id = coordinates_data[rec]["name"];
        for (var n = 1; n <= 4; ++ n){
            var src = imgLoc+"GSV_PILOT_"+loc_id+"_"+n+".jpg";
            var img =  $('<img />',{class:"img-thumbnail img_thumb_gal", id:"img_"+loc_id+"_"+n, alt:"img"+n, src:src});
            img.appendTo($("#gallery-content"))
        }
    }
}
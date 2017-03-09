// Place all the behaviors and hooks related to the matching controller here.
// All this logic will automatically be available in application.js.

var ans = "";
var affect_active = true;

function setCharAt(str,index,chr) {
    if(index > str.length-1) return str;
    alert(str+" "+index+" "+chr);
    return str.substr(0,index) + chr + str.substr(index+1);
}

$(function() {

    $("#start_time").ready(function(){
        $("#start_time").val(new Date($.now()));
    });

    //ACR scale related
    $("#slider-Attractiveness").slider({
        value: 3,
        min: 1,
        max: 5,
        step: 1,
        stop: function (event, ui) {
            $("#attractiveness").val(ui.value);
            $("#q2").removeAttr("hidden");
        }
    });

    $("#attractiveness").val($("#slider-Attractiveness").slider("value"));

    $("input[name='familiarity']").change(function(){
        $("#familiarity").val($(this).val());
        $("#q3").removeAttr("hidden");
});

    //ACR scale related
    $("#slider-Uniqueness").slider({
        value: 3,
        min: 1,
        max: 5,
        step: 1,
        stop: function (event, ui) {
            $("#uniqueness").val(ui.value);
            $("#q4").removeAttr("hidden");
        }
    });
    $("#uniqueness").val($("#slider-Uniqueness").slider("value"));

    $("input[name='friendliness']").change(function(){
        $("#friendliness").val($(this).val());
        $("#q5").removeAttr("hidden");
    });

    $("#golden_q").ready(function(){
        if($("#checkimage").val() == 1){
            $("#golden_q").append($("<hr>"));
            $("#golden_q").append($("<p>Which of the following <b>objects</b> appear in this image? </p>"));

            if($("#thisistraining").length!=0){
                $("#golden_q").append($("<i>Please check which objects appear in the image. You may choose more than one options.</i><br>"));
            }

            var options = $("#options").val();
            options = options.split("|");
            var i_op=1;
            options.forEach(function(op) {
                var oppp = $('<input />', { type: 'checkbox', class: 'golden', id: 'golden'+i_op, value: i_op });
                ans = ans+"0";
                oppp.click(function(){
                    if(oppp.is(":checked")){
                        ans = ans.substr(0,oppp.val()-1) + "1" + ans.substr(oppp.val());
                    }else{
                        ans = ans.substr(0,oppp.val()-1) + "0" + ans.substr(oppp.val());
                    }
                    $("#golden_answer").val("a"+ans);
                    if(!affect_active) {
                        $("#next_btn").removeAttr("hidden");
                    }
                });

                oppp.appendTo($("#golden_q"));
                $("#golden_q").append(op+"<br>");
                i_op++;
            });

            $("#part").val(0);
            $("#golden_answer").val("");
        }else{
            $("#part").val(1);
        }
    });


    $('#affect').affectbutton({
    }).bind('affectchanged', function(e, a) {
        // ... so we can update the input element of each component
        //var aff = "";
        $.each(a, function(c, v) {
            //aff = aff+"; "+c+"="+v;
            $('#' + c).val(v);
        });
        //alert(aff);

        $('#affect').affectbutton('alive',0);

        $("#affectchanger_div").removeAttr("hidden");
        if($("#checkimage").val() == 1){
            if( $("#golden_q").prop("hidden") ){
                $("#golden_q").removeAttr("hidden");
            }else{
                $("#next_btn").removeAttr("hidden");
            }
        }else{
            $("#next_btn").removeAttr("hidden");
        }
        affect_active = false;
    });

    $("#affectchanger").click(function(){
        $("#affect").affectbutton("reset");
        $("#affectchanger_div").prop("hidden",true);
        $("#next_btn").prop("hidden",true);
        affect_active = true;
    });

});
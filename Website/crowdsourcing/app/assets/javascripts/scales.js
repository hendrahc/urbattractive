// Place all the behaviors and hooks related to the matching controller here.
// All this logic will automatically be available in application.js.

var ans = "";

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
        $("#golden_q").removeAttr("hidden");
    });

    //ACR scale related
    $("#slider-SAM").slider({
        value: 3,
        min: 1,
        max: 5,
        step: 1,
        stop: function (event, ui) {
            $("#SAM").val(ui.value);
            $("#golden_q").removeAttr("hidden");
            $("#next_btn").removeAttr("hidden");
        }
    });
    $("#SAM").val($("#slider-SAM").slider("value"));

    $("#SAM1").click(function(){
        $("#slider-SAM").slider("value" , 1);
        $("#next_btn").removeAttr("hidden");
        $("#golden_q").removeAttr("hidden");
    });
    $("#SAM2").click(function(){
        $("#slider-SAM").slider("value" , 2);
        $("#next_btn").removeAttr("hidden");
        $("#golden_q").removeAttr("hidden");
    });
    $("#SAM3").click(function(){
        $("#slider-SAM").slider("value" , 3);
        $("#next_btn").removeAttr("hidden");
        $("#golden_q").removeAttr("hidden");
    });
    $("#SAM4").click(function(){
        $("#slider-SAM").slider("value" , 4);
        $("#next_btn").removeAttr("hidden");
        $("#golden_q").removeAttr("hidden");
    });
    $("#SAM5").click(function(){
        $("#slider-SAM").slider("value" , 5);
        $("#next_btn").removeAttr("hidden");
        $("#golden_q").removeAttr("hidden");
    });

    $("#golden1").click(function(){
        alert("hello");
        $("#next_btn").removeAttr("hidden");
    });

    $("#golden_q").ready(function(){
        if($("#checkimage").val() == 1){
            $("#golden_q").append($("<hr>"));
            $("#golden_q").append($("<p>Which <b>objects</b> are appeared in this image?</p>"));
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
                    $("#next_btn").removeAttr("hidden");
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


});
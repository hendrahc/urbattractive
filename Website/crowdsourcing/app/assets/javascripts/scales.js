// Place all the behaviors and hooks related to the matching controller here.
// All this logic will automatically be available in application.js.
$(function() {

    //ACR scale related
    $("#slider-Attractiveness").slider({
        value: 3,
        min: 1,
        max: 5,
        step: 1,
        slide: function (event, ui) {
            $("#attractiveness").val(ui.value);
        }
    });
    $("#attractiveness").val($("#slider-Attractiveness").slider("value"));

    //ACR scale related
    $("#slider-Uniqueness").slider({
        value: 3,
        min: 1,
        max: 5,
        step: 1,
        slide: function (event, ui) {
            $("#uniqueness").val(ui.value);
        }
    });
    $("#uniqueness").val($("#slider-Uniqueness").slider("value"));

    //ACR scale related
    $("#slider-SAM").slider({
        value: 3,
        min: 1,
        max: 5,
        step: 1,
        slide: function (event, ui) {
            $("#SAM").val(ui.value);
        }
    });
    $("#SAM").val($("#slider-SAM").slider("value"));

});
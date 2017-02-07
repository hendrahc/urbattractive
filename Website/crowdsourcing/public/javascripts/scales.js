// Place all the behaviors and hooks related to the matching controller here.
// All this logic will automatically be available in application.js.
$(function() {
    $( "#slider" ).slider({
      value:50,
      min: 0,
      max: 100,
      step: 1,
      slide: function( event, ui ) {
        $( "#amount" ).val( ui.value );
      }
    });
    $( "#amount" ).val( $( "#slider" ).slider( "value" ) );
  
  //ACR scale related
    $( "#slider-ACR" ).slider({
      value: 3,
      min: 1,
      max: 5,
      step: 1,
      slide: function( event, ui ) {
        $( "#amount" ).val( ui.value );
      }
    });
    $( "#amount" ).val( $( "#slider-ACR" ).slider( "value" ) );

      //ACR scale related
    $( "#slider-ACR-Rec" ).slider({
      value: 3,
      min: 1,
      max: 5,
      step: 1,
      slide: function( event, ui ) {
        $( "#amount-Rec" ).val( ui.value );
      }
    });
    $( "#amount-Rec" ).val( $( "#slider-ACR-Rec" ).slider( "value" ) );

    //7-point scale related
    $( "#slider-7pt" ).slider({
      value:40,
      min: 10,
      max: 70,
      step: 1,
      slide: function( event, ui ) {
        $( "#amount" ).val( ui.value );
      }
    });
    $( "#amount" ).val( $( "#slider-7pt" ).slider( "value" ) );

//checkbox related
    $("#chk").click(function() {
       $("#amount").val($(this).is(':checked') ? '1' : '0');}
    )


  });

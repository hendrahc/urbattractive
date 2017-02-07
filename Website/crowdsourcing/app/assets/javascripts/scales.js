// Place all the behaviors and hooks related to the matching controller here.
// All this logic will automatically be available in application.js.
$(function() {
  
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



  });

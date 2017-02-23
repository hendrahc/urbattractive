/*
|--------------------------------------------------------------------------
| Likert Rating Plugin
|
| Builds likert rating bars from form select elements.
|
| Author: Daniel Sparr, Là-bas AB
|
| Based on Quickie plugin by Marco Kuiper (http://www.marcofolio.net/)
|--------------------------------------------------------------------------
*/

(function( $ ) {
    
    $.fn.likert = function( options ) {
 
        var settings = $.extend({
            // These are the defaults.
            animationTime: 500,
            colors: ["ff0000", "ff9900", "ffff00", "99ff00", "00ff00"],
            backgroundColor: "#000",
            infoBox: true,
            before: '',
            after: '',
            selectCallback: function() {}
        }, options );

        var colourizeRatings = function(el, nrOfRatings) {
            $(".likert-bar li a", el).each(function() {
                if($(this).parent().index() <= nrOfRatings) {
                    $(this).stop().animate({ backgroundColor : "#" + settings.colors[nrOfRatings] } , settings.animationTime);
                } else {
                    $(this).stop().animate({ backgroundColor : settings.backgroundColor } , settings.animationTime);
                }
            });
        };

        return this.each(function() {

            var selectElement = this;
            $(this).wrap('<div class="likert-container" />');
            var likertContainer = $(this).parent(); 
            var likertInfoBoxHTML = settings.infoBox ? '<div class="likert-info-box"></div>' : '';
            var likertBar = $('<ul class="likert-bar"></ul>' + likertInfoBoxHTML + '</div><!-- .likert-bar -->').insertAfter(selectElement);

            $(selectElement).hide();

            $(selectElement).prop("selectedIndex", -1);
        
            // Make an undordered list from the option elements
            if (settings.before) {
                $('.likert-bar', likertContainer).append('<li>' + settings.before + '</li>');
            }
            $('option', selectElement).each(function( index, optionObject ) {
                var text = optionObject.text;
                var value = optionObject.value;
                $( '.likert-bar', likertContainer ).append('<li><a href="#">' + text + '</a></li>');
            });
            if (settings.after) {
                $('.likert-bar', likertContainer).append('<li>' + settings.after + '</li>');
            }

            $('.likert-bar li a', likertContainer ).css('backgroundColor', settings.backgroundColor);

            // Handle selection (click)
            $('li a', likertContainer).click(function(e) {
                e.preventDefault();
                index = $(this).parent().index();
                $(selectElement).prop("selectedIndex", index);
                settings.selectCallback(this);
            });

            // Handle the hover events
            $('.likert-bar li a', likertContainer).hover(function() {

                // Fade in likert info box
                $('.likert-info-box', likertContainer)
                    .empty()
                    .stop()
                    .animate({ opacity : 1 }, settings.animationTime);

                // Add text to likert info box
                $('.likert-info-box', likertContainer)
                    .html($(this).html());

                // Call the colourize function with the given index
                colourizeRatings(likertContainer, $(this).parent().index());

            }, function() {
        
                // Fade out the rating information box
                if ( $(selectElement).prop("selectedIndex") == -1 ) {
                    $('.likert-info-box', likertContainer)
                        .stop()
                        .animate({ opacity : 0 }, settings.animationTime);                    
                } else {
                    text = $('option:selected', selectElement).text();
                    $('.likert-info-box', likertContainer).html(text);
                }
      
                // Restore all the rating to their original colours
                $(".likert-bar li a", likertContainer).stop().animate({ backgroundColor : settings.backgroundColor } , settings.animationTime);
                index = $(selectElement).prop("selectedIndex");
                if (  index != -1 ) {
                    colourizeRatings(likertContainer, index);
                }
            });    

        });

    };
 
}( jQuery ));
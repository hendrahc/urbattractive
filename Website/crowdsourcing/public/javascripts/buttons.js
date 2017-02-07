seconds = 5;

function decreaseTime(){
  document.getElementById("start-button").innerHTML = "Wait " + seconds + " seconds";
  seconds--;
  if(seconds<0){
    document.getElementById("start-button").innerHTML = "Start now!";

    return true;
  }
  setTimeout('decreaseTime()',1500);
}

window.onload = function() {
  document.getElementById("start-button").innerHTML = "Wait " + seconds + " seconds" ;
  var e = document.getElementById("start-button");
  lockoutSubmit(e);
  decreaseTime();
}

function lockoutSubmit(button) {
    button.setAttribute('disabled', 'disabled');
    button.onclick = function() {return false; }
    setTimeout(function(){
        button.removeAttribute('disabled');
        button.onclick = function() {return true; }
    }, 7500)

}
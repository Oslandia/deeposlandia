// Jitenshea

// PREFIX should be defined in the settings.js file
var API_URL = PREFIX + '/api';

// Build the URL with a BASE_URL/<city> suffix based from a DOM element with the
// "city" dataset attribute, i.e. 'data-city'.
function cityurl(dom_id) {
  return API_URL + "/" + document.getElementById(dom_id).dataset.city;
};

// get the date before today in YYYY-MM-DD string format
function getYesterday() {
  var yesterday = new Date()
  yesterday.setDate(yesterday.getDate() - 1);
  // console.log(yesterday.toISOString().substring(0, 10));
  return yesterday.toISOString().substring(0, 10);
};


// Load a new image from the static directory on click
var predict_button = document.getElementById("predict_labels");
predict_button.addEventListener("click", function(){
  var image = document.getElementById("raw_image");

  console.log("Generate a new image...");
  filename = generate_image(image);
  document.getElementById("result").innerHTML = "";
  console.log(filename);

  console.log("Predict labels for " + model);
  predict_labels(filename, model);
});

function generate_image(image){
  if(image.complete){
    var new_image = new Image();
    new_image.id = "raw_image";
    var image_id_str = "" + Math.floor(Math.random() * 5000 + 1);
    var image_id = ('00000'+image_id_str).substring(image_id_str.length);
    filename =  "/static/images/shape_".concat(image_id, ".png");
    new_image.src = filename
    image.parentNode.insertBefore(new_image, image);
    image.parentNode.removeChild(image);
  }
  return filename
};

function predict_labels(filename, model){
  $.getJSON('/shape_prediction', {
    img: filename,
    model: model
  }, function(data){
    var result = [];
    if (model === "feature_detection") {
      $.each(data, function(image, predictions){
	result.push("<ul>");
	$.each(predictions, function(key, val){
      	  result.push("<li>" + key + ": " + val + "%</li>" );
	});
	result.push("</ul>");
      });
      document.getElementById("result").innerHTML = result.join("")
    } else {
      var predicted_image_path;
      $.each(data.lab_images, function(image, predicted_image){
	predicted_image_path = "/static/predicted_images/" + predicted_image;
	result.push("<img id='predicted_image'><label>Predicted labels</label>");
      });
      result.push("<ul>");
      $.each(data.labels, function(label, color){
	if (label != "background") {
    	  result.push("<li><font color='" + color + "'>" + label + "</font></li>" );
	}
      });
      result.push("</ul>");
      document.getElementById("result").innerHTML = result.join("");
      document.getElementById("predicted_image").src = predicted_image_path;
    }
  });/*$.getJSON*/
};

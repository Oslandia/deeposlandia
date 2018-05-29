// Load a new image from the static directory on click
var predict_button = document.getElementById("predict_labels");
predict_button.addEventListener("click", function(){
  var image = document.getElementById("image");

  console.log("Generate a new image...");
  filename = generate_image(image);
  document.getElementById("result").innerHTML = "";

  console.log("Predict labels...");
  predict_labels(filename, "shapes", "feature_detection");
});

function generate_image(image){
  if(image.complete){
    var new_image = new Image();
    new_image.id = "image";
    var image_id_str = "" + Math.floor(Math.random() * 5000 + 1);
    var image_id = ('00000'+image_id_str).substring(image_id_str.length);
    filename =  "/static/images/shape_".concat(image_id, ".png");
    new_image.src = filename
    image.parentNode.insertBefore(new_image, image);
    image.parentNode.removeChild(image);
  }
  return filename
};

function predict_labels(filename, dataset, model){
  $.getJSON('/shape_prediction', {
    img: filename,
    dataset: dataset,
    model: model
  }, function(data){
    var result = [];
    $.each(data, function(image, predictions){
      result.push("<ul>");
      $.each(predictions, function(key, val){
      	result.push("<li>" + key + ": " + val + "%</li>" );
      });
      result.push("</ul>");
      console.log(result.join(""))
    });
    document.getElementById("result").innerHTML = result.join("")
  });/*$.getJSON*/
};

// Load a new image from the static directory on click
document.getElementById("predict_button").addEventListener("click", function(){

  document.getElementById("result_label").innerHTML = "";
  console.log("Predict labels for " + model);
  var uploaded_image = document.getElementById("uploaded_image");
  var filename = uploaded_image.src;
  predict_labels(filename, "mapillary", model);
  console.log("Prediction OK!");

  function predict_labels(filename, dataset, model){

    $.getJSON(PREFIX + '/prediction', {
      img: filename,
      dataset: dataset,
      model: model
    }, function(data){
      var result = [];
      var predicted_image_path;
      $.each(data.label_images, function(image, predicted_image){
        predicted_image_path = "/static/predicted/" + predicted_image;
      });
      $.each(data.labels, function(id, label){
        if (label != "background") {
          result.push("<span class='color-label' style='background-color:" + label[1] + "'>" + label[0] + "</span>");
        }
      });
      console.log(result.join(""));
      document.getElementById("result_label").innerHTML = result.join("");
      document.getElementById("predictions").src = predicted_image_path;
    }); //$.getJSON
  };
});

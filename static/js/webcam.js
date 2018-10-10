var video = document.querySelector("#videoElement");
/*
const canvas = document.getElementById('#canvas');
const context = canvas.getContext('2d');
const constraints = {
  video: true,
};

navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    video.srcObject = stream;
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
  });
  *
////////////////////////////////////
Webcam.set('constraints', {
  width: 640,
  height: 480,
  image_format: 'jpeg',
  jpeg_quality: 90
});
Webcam.attach('#containerVideo');

var timer = null;

function take_snapshot() {
  // take snapshot and get image data
  Webcam.snap(function (data_uri) {
    // display results in page
    var img = new Image();
    img.src = data_uri;

    document.getElementById('results').appendChild(img);
  });
}

function start_snapping() {
  if (!timer) {
    take_snapshot();
    timer = setInterval(take_snapshot, 50);
  }
}

function stop_snapping() {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

function erase_snaps() {
  document.getElementById('results').innerHTML = '';
}

////////////////////////////////////
*
count = 0;
if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({video: true})
  .then(function(stream) {
    video.srcObject = stream;
    video.onloadedmetadata = function(e) {
      var img = capture(video);
      
    }
  })
  .catch(function(err0r) {
    console.log("Something went wrong!");
  });
}

capture  = function(video, scaleFactor) {
  //scaleFactor: optional param to resize the img if needed
  if(scaleFactor == null){
    scaleFactor = 1;
  } 
  var w = video.videoWidth * scaleFactor;
  var h = video.videoHeight * scaleFactor;
  var canvas = document.createElement('canvas');
  canvas.width  = w;
  canvas.height = h;
  var ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  frame_data = ctx.getImageData(0,0,w,h);
  frame = frame_data.data;
  img = canvas.toDataURL("image/jpg");
  return img;
} 
*/
function startPython() {
  var exec = require('child_process').exec;
  let pythonProcess = exec('python3 tf-pose-estimation/run_v2.py');
  console.log("Done");
  pythonProcess.stderr.on('data', (data) => {
    console.log(`stderr: ${data}`);
  });
}

var uint8arrayToString = function(data){
  return String.fromCharCode.apply(null, data);
};

function test() {

  var spawn = require('child_process').spawn;
  let pythonProcess = spawn('python3', ['test.py']);

  const constraints = {
    video: true,
  };

  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');

  navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    video.srcObject = stream;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  
    var imgData = canvas.toDataURL();
    var data = JSON.stringify(imgData);
    pythonProcess.stdin.write(data);
    pythonProcess.stdin.end();

    var image = new Image();
    image.onload = function() {
      ctx.drawImage(image, 0, 0);
    };

    pythonProcess.stdout.on('data', (data) => {
      var prefix = "data:image/png;base64,";
      var img = prefix + uint8arrayToString(data);
      image.src = img;
      console.log(img);
    });
    pythonProcess.stderr.on('data', (data) => {
      console.log(`stderr: ${data}`);
    });
  });
}

/*
console.clear();
;
(function () {

  navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia;

  if (!navigator.getUserMedia) {
    return false;
  }

  var width = 0,
    height = 0;

  var canvas = document.createElement('canvas'),
    ctx = canvas.getContext('2d');
  document.body.appendChild(canvas);

  var video = document.createElement('video'),
    track;
  video.setAttribute('autoplay', true);

  window.vid = video;

  function getWebcam() {

    navigator.getUserMedia({
      video: true,
      audio: false
    }, function (stream) {
      video.src = window.URL.createObjectURL(stream);
      track = stream.getTracks()[0];
    }, function (e) {
      console.error('Rejected!', e);
    });
  }

  getWebcam();

  var rotation = 0,
    loopFrame,
    centerX,
    centerY,
    twoPI = Math.PI * 2;

  function loop() {

    loopFrame = requestAnimationFrame(loop);
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx
    ctx.drawImage(video, 0, 0, width, height);

    ctx.restore();

  }

  function startLoop() {
    loopFrame = loopFrame || requestAnimationFrame(loop);
  }

  video.addEventListener('loadedmetadata', function () {
    width = canvas.width = video.videoWidth;
    height = canvas.height = video.videoHeight;
    centerX = width / 2;
    centerY = height / 2;
    startLoop();
  });

  canvas.addEventListener('click', function () {
    if (track) {
      if (track.stop) {
        track.stop();
      }
      track = null;
    } else {
      getWebcam();
    }
  });
})()
*/
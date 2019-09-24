<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<title>finding_donors</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    color: #000 !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.2.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.2.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.2.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.2.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.2.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.2.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=1);
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2);
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=3);
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1);
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1);
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
@media (max-width: 991px) {
  #ipython_notebook {
    margin-left: 10px;
  }
}
[dir="rtl"] #ipython_notebook {
  float: right !important;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#login_widget {
  float: right;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  text-align: center;
  vertical-align: middle;
  display: inline;
  opacity: 0;
  z-index: 2;
  width: 12ex;
  margin-right: -12ex;
}
.alternate_upload .btn-upload {
  height: 22px;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
[dir="rtl"] #tabs li {
  float: right;
}
ul#tabs {
  margin-bottom: 4px;
}
[dir="rtl"] ul#tabs {
  margin-right: 0px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons {
  float: left !important;
}
[dir="rtl"] .list_toolbar .pull-right {
  padding-top: 1px;
  float: left !important;
}
[dir="rtl"] .list_toolbar .pull-left {
  float: right !important;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: baseline;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
#tree-selector {
  padding-right: 0px;
}
[dir="rtl"] #tree-selector a {
  float: right;
}
#button-select-all {
  min-width: 50px;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
[dir="rtl"] #new-menu {
  text-align: right;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
[dir="rtl"] #running .col-sm-8 {
  float: right !important;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI colors. */
.ansibold {
  font-weight: bold;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  border-left-width: 1px;
  padding-left: 5px;
  background: linear-gradient(to right, transparent -40px, transparent 1px, transparent 1px, transparent 100%);
}
div.cell.jupyter-soft-selected {
  border-left-color: #90CAF9;
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected {
  border-color: #ababab;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 5px, transparent 5px, transparent 100%);
}
@media print {
  div.cell.selected {
    border-color: transparent;
  }
}
div.cell.selected.jupyter-soft-selected {
  border-left-width: 0;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 7px, #E3F2FD 7px, #E3F2FD 100%);
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #66BB6A -40px, #66BB6A 5px, transparent 5px, transparent 100%);
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  padding: 0.4em;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. We need the 0 value because of how we size */
  /* .CodeMirror-lines */
  padding: 0;
  border: 0;
  border-radius: 0;
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul {
  list-style: disc;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ul ul {
  list-style: square;
  margin: 0em 2em;
}
.rendered_html ul ul ul {
  list-style: circle;
  margin: 0em 2em;
}
.rendered_html ol {
  list-style: decimal;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
  margin: 0em 2em;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  background-color: #fff;
  color: #000;
  font-size: 100%;
  padding: 0px;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: 1px solid black;
  border-collapse: collapse;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  border: 1px solid black;
  border-collapse: collapse;
  margin: 1em 2em;
}
.rendered_html td,
.rendered_html th {
  text-align: left;
  vertical-align: middle;
  padding: 4px;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget {
  float: right !important;
  float: right;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  margin-top: 6px;
}
span.save_widget span.filename {
  height: 1em;
  line-height: 1em;
  padding: 3px;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  display: none;
}
.command-shortcut:before {
  content: "(command)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Supervised-Learning">Supervised Learning<a class="anchor-link" href="#Supervised-Learning">&#182;</a></h2><h2 id="Project:-Finding-Donors-for-CharityML">Project: Finding Donors for <em>CharityML</em><a class="anchor-link" href="#Project:-Finding-Donors-for-CharityML">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with <strong>'Implementation'</strong> in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a <code>'TODO'</code> statement. Please be sure to read the instructions carefully!</p>
<p>In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a <strong>'Question X'</strong> header. Carefully read each question and provide thorough answers in the following text boxes that begin with <strong>'Answer:'</strong>. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.</p>
<blockquote><p><strong>Note:</strong> Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the <strong>Shift + Enter</strong> keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Getting-Started">Getting Started<a class="anchor-link" href="#Getting-Started">&#182;</a></h2><p>In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features.</p>
<p>The dataset for this project originates from the <a href="https://archive.ics.uci.edu/ml/datasets/Census+Income">UCI Machine Learning Repository</a>. The datset was donated by Ron Kohavi and Barry Becker, after being published in the article <em>"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"</em>. You can find the article by Ron Kohavi <a href="https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf">online</a>. The data we investigate here consists of small changes to the original dataset, such as removing the <code>'fnlwgt'</code> feature and records with missing or ill-formatted entries.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Exploring-the-Data">Exploring the Data<a class="anchor-link" href="#Exploring-the-Data">&#182;</a></h2><p>Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, <code>'income'</code>, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import libraries necessary for this project</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">from</span> <span class="nn">time</span> <span class="k">import</span> <span class="n">time</span>
<span class="kn">from</span> <span class="nn">IPython.display</span> <span class="k">import</span> <span class="n">display</span> <span class="c1"># Allows the use of display() for DataFrames</span>

<span class="c1"># Import supplementary visualization code visuals.py</span>
<span class="kn">import</span> <span class="nn">visuals</span> <span class="k">as</span> <span class="nn">vs</span>

<span class="c1"># Pretty display for notebooks</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="c1"># Load the Census dataset</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;census.csv&quot;</span><span class="p">)</span>

<span class="c1"># Success - Display the first record</span>
<span class="n">display</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">n</span><span class="o">=</span><span class="mi">1</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># data.income.unique()</span>
<span class="c1"># data.shape[0]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># a=data.loc[data[&#39;income&#39;]==&#39;&lt;=50K&#39;].shape[0]</span>
<span class="c1"># b=data.loc[data[&#39;income&#39;]==&#39;&gt;50K&#39;].shape[0]</span>
<span class="c1"># display(a,&quot;\t&quot;,b,&quot;\t&quot;,a+b)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Implementation:-Data-Exploration">Implementation: Data Exploration<a class="anchor-link" href="#Implementation:-Data-Exploration">&#182;</a></h3><p>A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, you will need to compute the following:</p>
<ul>
<li>The total number of records, <code>'n_records'</code></li>
<li>The number of individuals making more than \$50,000 annually, <code>'n_greater_50k'</code>.</li>
<li>The number of individuals making at most \$50,000 annually, <code>'n_at_most_50k'</code>.</li>
<li>The percentage of individuals making more than \$50,000 annually, <code>'greater_percent'</code>.</li>
</ul>
<p><strong> HINT: </strong> You may need to look at the table above to understand how the <code>'income'</code> entries are formatted.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># TODO: Total number of records</span>
<span class="n">n_records</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># TODO: Number of records where individual&#39;s income is more than $50,000</span>
<span class="n">n_greater_50k</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">&#39;income&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;&gt;50K&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># TODO: Number of records where individual&#39;s income is at most $50,000</span>
<span class="n">n_at_most_50k</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">&#39;income&#39;</span><span class="p">]</span><span class="o">==</span><span class="s1">&#39;&lt;=50K&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># TODO: Percentage of individuals whose income is more than $50,000</span>
<span class="n">greater_percent</span> <span class="o">=</span> <span class="p">(</span><span class="n">n_greater_50k</span> <span class="o">/</span> <span class="n">n_records</span> <span class="p">)</span> <span class="o">*</span> <span class="mi">100</span>

<span class="c1"># Print the results</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Total number of records: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">n_records</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Individuals making more than $50,000: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">n_greater_50k</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Individuals making at most $50,000: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">n_at_most_50k</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Percentage of individuals making more than $50,000: </span><span class="si">{}</span><span class="s2">%&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">greater_percent</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Total number of records: 45222
Individuals making more than $50,000: 11208
Individuals making at most $50,000: 34014
Percentage of individuals making more than $50,000: 24.78439697492371%
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong> Featureset Exploration </strong></p>
<ul>
<li><strong>age</strong>: continuous. </li>
<li><strong>workclass</strong>: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. </li>
<li><strong>education</strong>: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. </li>
<li><strong>education-num</strong>: continuous. </li>
<li><strong>marital-status</strong>: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. </li>
<li><strong>occupation</strong>: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. </li>
<li><strong>relationship</strong>: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. </li>
<li><strong>race</strong>: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. </li>
<li><strong>sex</strong>: Female, Male. </li>
<li><strong>capital-gain</strong>: continuous. </li>
<li><strong>capital-loss</strong>: continuous. </li>
<li><strong>hours-per-week</strong>: continuous. </li>
<li><strong>native-country</strong>: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&amp;Tobago, Peru, Hong, Holand-Netherlands.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Preparing-the-Data">Preparing the Data<a class="anchor-link" href="#Preparing-the-Data">&#182;</a></h2><p>Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as <strong>preprocessing</strong>. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Transforming-Skewed-Continuous-Features">Transforming Skewed Continuous Features<a class="anchor-link" href="#Transforming-Skewed-Continuous-Features">&#182;</a></h3><p>A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '<code>capital-gain'</code> and <code>'capital-loss'</code>.</p>
<p>Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Split the data into features and target label</span>
<span class="n">income_raw</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">&#39;income&#39;</span><span class="p">]</span>
<span class="n">features_raw</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;income&#39;</span><span class="p">,</span> <span class="n">axis</span> <span class="o">=</span> <span class="mi">1</span><span class="p">)</span>

<span class="c1"># Visualize skewed continuous features of original data</span>
<span class="n">vs</span><span class="o">.</span><span class="n">distribution</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For highly-skewed feature distributions such as <code>'capital-gain'</code> and <code>'capital-loss'</code>, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of <code>0</code> is undefined, so we must translate the values by a small amount above <code>0</code> to apply the the logarithm successfully.</p>
<p>Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Log-transform the skewed features</span>
<span class="n">skewed</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;capital-gain&#39;</span><span class="p">,</span> <span class="s1">&#39;capital-loss&#39;</span><span class="p">]</span>
<span class="n">features_log_transformed</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">data</span> <span class="o">=</span> <span class="n">features_raw</span><span class="p">)</span>
<span class="n">features_log_transformed</span><span class="p">[</span><span class="n">skewed</span><span class="p">]</span> <span class="o">=</span> <span class="n">features_raw</span><span class="p">[</span><span class="n">skewed</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">np</span><span class="o">.</span><span class="n">log</span><span class="p">(</span><span class="n">x</span> <span class="o">+</span> <span class="mi">1</span><span class="p">))</span>

<span class="c1"># Visualize the new log distributions</span>
<span class="n">vs</span><span class="o">.</span><span class="n">distribution</span><span class="p">(</span><span class="n">features_log_transformed</span><span class="p">,</span> <span class="n">transformed</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Normalizing-Numerical-Features">Normalizing Numerical Features<a class="anchor-link" href="#Normalizing-Numerical-Features">&#182;</a></h3><p>In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as <code>'capital-gain'</code> or <code>'capital-loss'</code> above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.</p>
<p>Run the code cell below to normalize each numerical feature. We will use <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html"><code>sklearn.preprocessing.MinMaxScaler</code></a> for this.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import sklearn.preprocessing.StandardScaler</span>
<span class="kn">from</span> <span class="nn">sklearn.preprocessing</span> <span class="k">import</span> <span class="n">MinMaxScaler</span>

<span class="c1"># Initialize a scaler, then apply it to the features</span>
<span class="n">scaler</span> <span class="o">=</span> <span class="n">MinMaxScaler</span><span class="p">()</span> <span class="c1"># default=(0, 1)</span>
<span class="n">numerical</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;age&#39;</span><span class="p">,</span> <span class="s1">&#39;education-num&#39;</span><span class="p">,</span> <span class="s1">&#39;capital-gain&#39;</span><span class="p">,</span> <span class="s1">&#39;capital-loss&#39;</span><span class="p">,</span> <span class="s1">&#39;hours-per-week&#39;</span><span class="p">]</span>

<span class="n">features_log_minmax_transform</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">data</span> <span class="o">=</span> <span class="n">features_log_transformed</span><span class="p">)</span>
<span class="n">features_log_minmax_transform</span><span class="p">[</span><span class="n">numerical</span><span class="p">]</span> <span class="o">=</span> <span class="n">scaler</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">features_log_transformed</span><span class="p">[</span><span class="n">numerical</span><span class="p">])</span>

<span class="c1"># Show an example of a record with scaling applied</span>
<span class="n">display</span><span class="p">(</span><span class="n">features_log_minmax_transform</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">n</span> <span class="o">=</span> <span class="mi">5</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.301370</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.667492</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.452055</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.122449</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.287671</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>0.533333</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.493151</td>
      <td>Private</td>
      <td>11th</td>
      <td>0.400000</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.150685</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>Cuba</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Implementation:-Data-Preprocessing">Implementation: Data Preprocessing<a class="anchor-link" href="#Implementation:-Data-Preprocessing">&#182;</a></h3><p>From the table in <strong>Exploring the Data</strong> above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called <em>categorical variables</em>) be converted. One popular way to convert categorical variables is by using the <strong>one-hot encoding</strong> scheme. One-hot encoding creates a <em>"dummy"</em> variable for each possible category of each non-numeric feature. For example, assume <code>someFeature</code> has three possible entries: <code>A</code>, <code>B</code>, or <code>C</code>. We then encode this feature into <code>someFeature_A</code>, <code>someFeature_B</code> and <code>someFeature_C</code>.</p>
<table>
<thead><tr>
<th style="text-align:center"></th>
<th style="text-align:center">someFeature</th>
<th></th>
<th style="text-align:center">someFeature_A</th>
<th style="text-align:center">someFeature_B</th>
<th style="text-align:center">someFeature_C</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">0</td>
<td style="text-align:center">B</td>
<td></td>
<td style="text-align:center">0</td>
<td style="text-align:center">1</td>
<td style="text-align:center">0</td>
</tr>
<tr>
<td style="text-align:center">1</td>
<td style="text-align:center">C</td>
<td>----&gt; one-hot encode ----&gt;</td>
<td style="text-align:center">0</td>
<td style="text-align:center">0</td>
<td style="text-align:center">1</td>
</tr>
<tr>
<td style="text-align:center">2</td>
<td style="text-align:center">A</td>
<td></td>
<td style="text-align:center">1</td>
<td style="text-align:center">0</td>
<td style="text-align:center">0</td>
</tr>
</tbody>
</table>
<p>Additionally, as with the non-numeric features, we need to convert the non-numeric target label, <code>'income'</code> to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("&lt;=50K" and "&gt;50K"), we can avoid using one-hot encoding and simply encode these two categories as <code>0</code> and <code>1</code>, respectively. In code cell below, you will need to implement the following:</p>
<ul>
<li>Use <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies"><code>pandas.get_dummies()</code></a> to perform one-hot encoding on the <code>'features_log_minmax_transform'</code> data.</li>
<li>Convert the target label <code>'income_raw'</code> to numerical entries.<ul>
<li>Set records with "&lt;=50K" to <code>0</code> and records with "&gt;50K" to <code>1</code>.</li>
</ul>
</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">income_to_num</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="k">if</span> <span class="n">x</span><span class="o">==</span><span class="s1">&#39;&gt;50K&#39;</span><span class="p">:</span>
        <span class="k">return</span> <span class="mi">1</span>
    <span class="k">if</span> <span class="n">x</span><span class="o">==</span><span class="s1">&#39;&lt;=50K&#39;</span><span class="p">:</span>
        <span class="k">return</span> <span class="mi">0</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># TODO: One-hot encode the &#39;features_log_minmax_transform&#39; data using pandas.get_dummies()</span>
<span class="n">features_final</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">features_log_minmax_transform</span><span class="p">)</span>

<span class="c1"># TODO: Encode the &#39;income_raw&#39; data to numerical values</span>
<span class="n">income</span> <span class="o">=</span> <span class="n">income_raw</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">income_to_num</span><span class="p">)</span>

<span class="c1"># Print the number of features after one-hot encoding</span>
<span class="n">encoded</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="n">features_final</span><span class="o">.</span><span class="n">columns</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{}</span><span class="s2"> total features after one-hot encoding.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">encoded</span><span class="p">)))</span>

<span class="c1"># Uncomment the following line to see the encoded feature names</span>
<span class="nb">print</span><span class="p">(</span><span class="n">encoded</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>103 total features after one-hot encoding.
[&#39;age&#39;, &#39;education-num&#39;, &#39;capital-gain&#39;, &#39;capital-loss&#39;, &#39;hours-per-week&#39;, &#39;workclass_ Federal-gov&#39;, &#39;workclass_ Local-gov&#39;, &#39;workclass_ Private&#39;, &#39;workclass_ Self-emp-inc&#39;, &#39;workclass_ Self-emp-not-inc&#39;, &#39;workclass_ State-gov&#39;, &#39;workclass_ Without-pay&#39;, &#39;education_level_ 10th&#39;, &#39;education_level_ 11th&#39;, &#39;education_level_ 12th&#39;, &#39;education_level_ 1st-4th&#39;, &#39;education_level_ 5th-6th&#39;, &#39;education_level_ 7th-8th&#39;, &#39;education_level_ 9th&#39;, &#39;education_level_ Assoc-acdm&#39;, &#39;education_level_ Assoc-voc&#39;, &#39;education_level_ Bachelors&#39;, &#39;education_level_ Doctorate&#39;, &#39;education_level_ HS-grad&#39;, &#39;education_level_ Masters&#39;, &#39;education_level_ Preschool&#39;, &#39;education_level_ Prof-school&#39;, &#39;education_level_ Some-college&#39;, &#39;marital-status_ Divorced&#39;, &#39;marital-status_ Married-AF-spouse&#39;, &#39;marital-status_ Married-civ-spouse&#39;, &#39;marital-status_ Married-spouse-absent&#39;, &#39;marital-status_ Never-married&#39;, &#39;marital-status_ Separated&#39;, &#39;marital-status_ Widowed&#39;, &#39;occupation_ Adm-clerical&#39;, &#39;occupation_ Armed-Forces&#39;, &#39;occupation_ Craft-repair&#39;, &#39;occupation_ Exec-managerial&#39;, &#39;occupation_ Farming-fishing&#39;, &#39;occupation_ Handlers-cleaners&#39;, &#39;occupation_ Machine-op-inspct&#39;, &#39;occupation_ Other-service&#39;, &#39;occupation_ Priv-house-serv&#39;, &#39;occupation_ Prof-specialty&#39;, &#39;occupation_ Protective-serv&#39;, &#39;occupation_ Sales&#39;, &#39;occupation_ Tech-support&#39;, &#39;occupation_ Transport-moving&#39;, &#39;relationship_ Husband&#39;, &#39;relationship_ Not-in-family&#39;, &#39;relationship_ Other-relative&#39;, &#39;relationship_ Own-child&#39;, &#39;relationship_ Unmarried&#39;, &#39;relationship_ Wife&#39;, &#39;race_ Amer-Indian-Eskimo&#39;, &#39;race_ Asian-Pac-Islander&#39;, &#39;race_ Black&#39;, &#39;race_ Other&#39;, &#39;race_ White&#39;, &#39;sex_ Female&#39;, &#39;sex_ Male&#39;, &#39;native-country_ Cambodia&#39;, &#39;native-country_ Canada&#39;, &#39;native-country_ China&#39;, &#39;native-country_ Columbia&#39;, &#39;native-country_ Cuba&#39;, &#39;native-country_ Dominican-Republic&#39;, &#39;native-country_ Ecuador&#39;, &#39;native-country_ El-Salvador&#39;, &#39;native-country_ England&#39;, &#39;native-country_ France&#39;, &#39;native-country_ Germany&#39;, &#39;native-country_ Greece&#39;, &#39;native-country_ Guatemala&#39;, &#39;native-country_ Haiti&#39;, &#39;native-country_ Holand-Netherlands&#39;, &#39;native-country_ Honduras&#39;, &#39;native-country_ Hong&#39;, &#39;native-country_ Hungary&#39;, &#39;native-country_ India&#39;, &#39;native-country_ Iran&#39;, &#39;native-country_ Ireland&#39;, &#39;native-country_ Italy&#39;, &#39;native-country_ Jamaica&#39;, &#39;native-country_ Japan&#39;, &#39;native-country_ Laos&#39;, &#39;native-country_ Mexico&#39;, &#39;native-country_ Nicaragua&#39;, &#39;native-country_ Outlying-US(Guam-USVI-etc)&#39;, &#39;native-country_ Peru&#39;, &#39;native-country_ Philippines&#39;, &#39;native-country_ Poland&#39;, &#39;native-country_ Portugal&#39;, &#39;native-country_ Puerto-Rico&#39;, &#39;native-country_ Scotland&#39;, &#39;native-country_ South&#39;, &#39;native-country_ Taiwan&#39;, &#39;native-country_ Thailand&#39;, &#39;native-country_ Trinadad&amp;Tobago&#39;, &#39;native-country_ United-States&#39;, &#39;native-country_ Vietnam&#39;, &#39;native-country_ Yugoslavia&#39;]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Shuffle-and-Split-Data">Shuffle and Split Data<a class="anchor-link" href="#Shuffle-and-Split-Data">&#182;</a></h3><p>Now all <em>categorical variables</em> have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.</p>
<p>Run the code cell below to perform this split.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="c1">#Changed it to model_selection to avoid deprecation warning</span>

<span class="c1"># Split the &#39;features&#39; and &#39;income&#39; data into training and testing sets</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">features_final</span><span class="p">,</span> 
                                                    <span class="n">income</span><span class="p">,</span> 
                                                    <span class="n">test_size</span> <span class="o">=</span> <span class="mf">0.2</span><span class="p">,</span> 
                                                    <span class="n">random_state</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>

<span class="c1"># Show the results of the split</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training set has </span><span class="si">{}</span><span class="s2"> samples.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Testing set has </span><span class="si">{}</span><span class="s2"> samples.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">X_test</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Training set has 36177 samples.
Testing set has 9045 samples.
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Evaluating-Model-Performance">Evaluating Model Performance<a class="anchor-link" href="#Evaluating-Model-Performance">&#182;</a></h2><p>In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a <em>naive predictor</em>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Metrics-and-the-Naive-Predictor">Metrics and the Naive Predictor<a class="anchor-link" href="#Metrics-and-the-Naive-Predictor">&#182;</a></h3><p><em>CharityML</em>, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using <strong>accuracy</strong> as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that <em>does not</em> make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is <em>more important</em> than the model's ability to <strong>recall</strong> those individuals. We can use <strong>F-beta score</strong> as a metric that considers both precision and recall:</p>
<p>$$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$</p>
<p>In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the <strong>F$_{0.5}$ score</strong> (or F-score for simplicity).</p>
<p>Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect <strong>accuracy</strong>, since we could simply say <em>"this person does not make more than \$50,000"</em> and generally be right, without ever looking at the data! Making such a statement would be called <strong>naive</strong>, since we have not considered any information to substantiate the claim. It is always important to consider the <em>naive prediction</em> for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, <em>CharityML</em> would identify no one as donors.</p>
<h4 id="Note:-Recap-of-accuracy,-precision,-recall">Note: Recap of accuracy, precision, recall<a class="anchor-link" href="#Note:-Recap-of-accuracy,-precision,-recall">&#182;</a></h4><p><strong> Accuracy </strong> measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).</p>
<p><strong> Precision </strong> tells us what proportion of messages we classified as spam, actually were spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of</p>
<p><code>[True Positives/(True Positives + False Positives)]</code></p>
<p><strong> Recall(sensitivity)</strong> tells us what proportion of messages that actually were spam were classified by us as spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of</p>
<p><code>[True Positives/(True Positives + False Negatives)]</code></p>
<p>For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score(we take the harmonic mean as we are dealing with ratios).</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-1---Naive-Predictor-Performace">Question 1 - Naive Predictor Performace<a class="anchor-link" href="#Question-1---Naive-Predictor-Performace">&#182;</a></h3><ul>
<li>If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? You must use the code cell below and assign your results to <code>'accuracy'</code> and <code>'fscore'</code> to be used later.</li>
</ul>
<p><strong> Please note </strong> that the the purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. In the real world, ideally your base model would be either the results of a previous model or could be based on a research paper upon which you are looking to improve. When there is no benchmark model set, getting a result better than random choice is a place you could start from.</p>
<p><strong> HINT: </strong></p>
<ul>
<li>When we have a model that always predicts '1' (i.e. the individual makes more than 50k) then our model will have no True Negatives(TN) or False Negatives(FN) as we are not making any negative('0' value) predictions. Therefore our Accuracy in this case becomes the same as our Precision(True Positives/(True Positives + False Positives)) as every prediction that we have made with value '1' that should have '0' becomes a False Positive; therefore our denominator in this case is the total number of records we have in total. </li>
<li>Our Recall score(True Positives/(True Positives + False Negatives)) in this setting becomes 1 as we have no False Negatives.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">TP</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">income</span><span class="p">)</span> <span class="c1"># Counting the ones as this is the naive case. Note that &#39;income&#39; is the &#39;income_raw&#39; data </span>
<span class="c1"># encoded to numerical values done in the data preprocessing step.</span>
<span class="n">FP</span> <span class="o">=</span> <span class="n">income</span><span class="o">.</span><span class="n">count</span><span class="p">()</span> <span class="o">-</span> <span class="n">TP</span> <span class="c1"># Specific to the naive case</span>

<span class="n">TN</span> <span class="o">=</span> <span class="mi">0</span> <span class="c1"># No predicted negatives in the naive case</span>
<span class="n">FN</span> <span class="o">=</span> <span class="mi">0</span> <span class="c1"># No predicted negatives in the naive case</span>

<span class="c1"># TODO: Calculate accuracy, precision and recall</span>
<span class="n">accuracy</span> <span class="o">=</span> <span class="n">TP</span> <span class="o">/</span> <span class="n">income</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
<span class="n">recall</span> <span class="o">=</span> <span class="mi">1</span>
<span class="n">precision</span> <span class="o">=</span> <span class="n">accuracy</span>

<span class="c1"># TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.</span>
<span class="n">beta</span> <span class="o">=</span> <span class="mf">0.5</span>
<span class="n">betasq</span> <span class="o">=</span> <span class="n">beta</span> <span class="o">*</span> <span class="n">beta</span>
<span class="n">fscore</span> <span class="o">=</span> <span class="p">(</span> <span class="p">(</span><span class="mi">1</span><span class="o">+</span><span class="n">betasq</span><span class="p">)</span> <span class="o">*</span> <span class="n">precision</span> <span class="o">*</span> <span class="n">recall</span><span class="p">)</span><span class="o">/</span> <span class="p">(</span> <span class="p">(</span><span class="n">betasq</span> <span class="o">*</span> <span class="n">precision</span><span class="p">)</span> <span class="o">+</span> <span class="n">recall</span> <span class="p">)</span>

<span class="c1"># Print the results </span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Naive Predictor: [Accuracy score: </span><span class="si">{:.4f}</span><span class="s2">, F-score: </span><span class="si">{:.4f}</span><span class="s2">]&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">accuracy</span><span class="p">,</span> <span class="n">fscore</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Supervised-Learning-Models">Supervised Learning Models<a class="anchor-link" href="#Supervised-Learning-Models">&#182;</a></h3><p><strong>The following are some of the supervised learning models that are currently available in</strong> <a href="http://scikit-learn.org/stable/supervised_learning.html"><code>scikit-learn</code></a> <strong>that you may choose from:</strong></p>
<ul>
<li>Gaussian Naive Bayes (GaussianNB)</li>
<li>Decision Trees</li>
<li>Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)</li>
<li>K-Nearest Neighbors (KNeighbors)</li>
<li>Stochastic Gradient Descent Classifier (SGDC)</li>
<li>Support Vector Machines (SVM)</li>
<li>Logistic Regression</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-2---Model-Application">Question 2 - Model Application<a class="anchor-link" href="#Question-2---Model-Application">&#182;</a></h3><p>List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen</p>
<ul>
<li>Describe one real-world application in industry where the model can be applied. </li>
<li>What are the strengths of the model; when does it perform well?</li>
<li>What are the weaknesses of the model; when does it perform poorly?</li>
<li>What makes this model a good candidate for the problem, given what you know about the data?</li>
</ul>
<p><strong> HINT: </strong></p>
<p>Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer: </strong></p>
<p><strong>1. Gaussian Naive Bayes</strong></p>
<p>1) <strong><em>Real World Example</em></strong>[1]</p>
<p>Classify a person’s cognitive activity, based on brain image</p>

<pre><code>    • reading a sentence or viewing a picture?

    • reading the word “Hammer” or “Apartment”

    • viewing a vertical or horizontal line?

    • answering the question, or getting confused?

</code></pre>
<p>2) <strong><em>Strengths</em></strong> [3]:</p>
<p>It's very simple and also if the NaiveBayes condition holds the classifier will converge quicker than other algorithms like Logistic Regression and even if the condition doesn't hold it still performs well.</p>
<p>3) <strong><em>Weaknesses</em></strong>[2]: 
   NaiveBayes models are often beaten by models properly trained and tuned using other algorithms.</p>
<p>4) <strong><em>Reason</em></strong>:</p>
<p>The reasons for choosing this model based on characterstics of dataset are as follows:</p>
<ul>
<li>The dataset contains more than 50 samples but less than 100K samples</li>
<li><p>Our dataset has labeled data, i.e. '&gt;50k' or '&lt;=50K'</p>
<p>All the above characteristics indicate using Naive Bayes algorithm for training according to the machine learning map given by scikit-learn[6].</p>
<p>Also we are unaware about the distribution of data points, so if the data is linearly separable, naive bayes will accurately separte the data. While i will choose the other two model such that they are good in separting non-linear data.</p>
</li>
</ul>
<p>The reasons for choosing this model based on characterstics of model are as follows:</p>
<ul>
<li><strong>Time</strong>:
   It takes less training time, as it classfies based on a simple condition.</li>
<li><strong>Accuracy</strong>:
   If the Naive Bayes condition holds it generally separates data efficiently but even if the condition doesn't hold up it still performs well.</li>
<li><strong>No of parameters</strong>:
   As per my knowledge, there are no parameters required for this model</li>
</ul>
<p><strong>2. Support Vector Machines </strong></p>
<p>1) <strong><em>Real World Example</em></strong>[4]:</p>
<p>Handwriting Recognition - SVMs are widely used to recognize handwritten characters</p>
<p>2) <strong><em>Strengths</em></strong>[2]:</p>
<p>SVM's can model non-linear decision boundaries. They are robust against overfitting even in high dimensional spaces.</p>
<p>3) <strong><em>Weaknesses</em></strong>[2]:</p>
<p>It is memory intensive, difficult to tune and doesn't scale well to large dataset.</p>
<p>4) <strong><em>Reason</em></strong>:</p>
<p>According to the machine learning map of scikit learn[6] discussed above SVC can also be used as an alternative.</p>
<p>Now, considering the model's charactestics which made me choose this model are as follows:</p>
<ul>
<li><strong>Linearity</strong>:
This algorithm has an ability to perform well with non-linear classification problems and as there as a lot of features(103 after one hot encoding), there is a high possibilty that the problem goes in n dimensional plane.</li>
<li><strong>No of Parameters</strong>:
There are number of parameters to tune like kernels, C value, gamma value etc., but once we get to know the right parameters, it can greatly affect the accuracy of the model. And in order to tune the parameters we can use GridSearch.</li>
<li><strong>Time</strong>:
SVC's take a lot of time in training especially with large data, but still i chose this model beacuse i wanted to compare all performance from all  categories of models. Nd this model belongs to the one, which needs long training time but delivers high accuracy.</li>
<li><strong>Accuracy</strong>:
This provides accuracy but at the cost of training performance.</li>
</ul>
<p><strong>3. Random Forests</strong></p>
<p>1) <strong><em>Real World Example</em></strong>[5]</p>
<p>In Finance, it is used to determine a stock's behaviour in the future.</p>
<p>2) <strong><em>Strengths</em></strong>[5]:</p>
<p>Random Forests are handy and easy to use. The default parameters often give a good prediction result and also the hyperparameters required is not that high. They prevent overfitting most of the time. It also helps to determine relative importance of features.</p>
<p>3) <strong><em>Weaknesses</em></strong>[5]:</p>
<p>Large number of trees can make the algorithm slow. Therefore cannot be used in real-time predictions.</p>
<p>4) <strong><em>Reason</em></strong></p>
<p>The last alternative suggested by scikit learn[6] in their machine learning map for having characteristics as discussed above(in GNB), is <em>Ensemble methods</em> and Random Forest is one of the ensemble methods.</p>
<p>And based on the model's characteristics they are as follows:</p>
<ul>
<li><strong>Avoids Overfitting</strong>:
These algorithm trains several smaller decision trees(weak learners) which are trained on limited knowledge. And then  these weak learners' output is combined(by averaging) to form a strong learner. This averaging helps the model to avoid overfitting over a large data as is the case in our dataset.</li>
<li><strong>Linearity</strong>:
This algorithm can classify non-linear datapoints which may be the case due to larger no. of features.</li>
<li><strong>No of parameters</strong>:
The model can be tuned according to a number of parameters such as no of estimators, criterion, maximum no of features etc.</li>
</ul>
<p>References:</p>
<ol>
<li>www.cs.cmu.edu/~tom/10601_sp09/lectures/NBayes2_2-2-2009-ann.pdf</li>
<li><a href="https://elitedatascience.com/machine-learning-algorithms">https://elitedatascience.com/machine-learning-algorithms</a></li>
<li><a href="https://www.cnblogs.com/yymn/p/4518016.html">https://www.cnblogs.com/yymn/p/4518016.html</a></li>
<li><a href="https://data-flair.training/blogs/applications-of-svm/">https://data-flair.training/blogs/applications-of-svm/</a></li>
<li><a href="https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd">https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd</a></li>
<li><a href="http://scikit-learn.org/stable/tutorial/machine_learning_map/">http://scikit-learn.org/stable/tutorial/machine_learning_map/</a></li>
</ol>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Implementation---Creating-a-Training-and-Predicting-Pipeline">Implementation - Creating a Training and Predicting Pipeline<a class="anchor-link" href="#Implementation---Creating-a-Training-and-Predicting-Pipeline">&#182;</a></h3><p>To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
In the code block below, you will need to implement the following:</p>
<ul>
<li>Import <code>fbeta_score</code> and <code>accuracy_score</code> from <a href="http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics"><code>sklearn.metrics</code></a>.</li>
<li>Fit the learner to the sampled training data and record the training time.</li>
<li>Perform predictions on the test data <code>X_test</code>, and also on the first 300 training points <code>X_train[:300]</code>.<ul>
<li>Record the total prediction time.</li>
</ul>
</li>
<li>Calculate the accuracy score for both the training subset and testing set.</li>
<li>Calculate the F-score for both the training subset and testing set.<ul>
<li>Make sure that you set the <code>beta</code> parameter!</li>
</ul>
</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># TODO: Import two metrics from sklearn - fbeta_score and accuracy_score</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">fbeta_score</span><span class="p">,</span> <span class="n">accuracy_score</span>
 
<span class="k">def</span> <span class="nf">train_predict</span><span class="p">(</span><span class="n">learner</span><span class="p">,</span> <span class="n">sample_size</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">):</span> 
    <span class="sd">&#39;&#39;&#39;</span>
<span class="sd">    inputs:</span>
<span class="sd">       - learner: the learning algorithm to be trained and predicted on</span>
<span class="sd">       - sample_size: the size of samples (number) to be drawn from training set</span>
<span class="sd">       - X_train: features training set</span>
<span class="sd">       - y_train: income training set</span>
<span class="sd">       - X_test: features testing set</span>
<span class="sd">       - y_test: income testing set</span>
<span class="sd">    &#39;&#39;&#39;</span>
    
    <span class="n">results</span> <span class="o">=</span> <span class="p">{}</span>
    
    <span class="c1"># TODO: Fit the learner to the training data using slicing with &#39;sample_size&#39; using .fit(training_features[:], training_labels[:])</span>
    <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span> <span class="c1"># Get start time</span>
    <span class="n">learner</span> <span class="o">=</span> <span class="n">learner</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">[:</span><span class="n">sample_size</span><span class="p">],</span><span class="n">y_train</span><span class="p">[:</span><span class="n">sample_size</span><span class="p">])</span>
    <span class="n">end</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span> <span class="c1"># Get end time</span>
    
    <span class="c1"># TODO: Calculate the training time</span>
    <span class="n">results</span><span class="p">[</span><span class="s1">&#39;train_time&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">end</span> <span class="o">-</span> <span class="n">start</span>
        
    <span class="c1"># TODO: Get the predictions on the test set(X_test),</span>
    <span class="c1">#       then get predictions on the first 300 training samples(X_train) using .predict()</span>
    <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span> <span class="c1"># Get start time</span>
    <span class="n">predictions_test</span> <span class="o">=</span> <span class="n">learner</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
    <span class="n">predictions_train</span> <span class="o">=</span> <span class="n">learner</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train</span><span class="p">[:</span><span class="mi">300</span><span class="p">])</span>
    <span class="n">end</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span> <span class="c1"># Get end time</span>
    
    <span class="c1"># TODO: Calculate the total prediction time</span>
    <span class="n">results</span><span class="p">[</span><span class="s1">&#39;pred_time&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">end</span> <span class="o">-</span> <span class="n">start</span>
            
    <span class="c1"># TODO: Compute accuracy on the first 300 training samples which is y_train[:300]</span>
    <span class="n">results</span><span class="p">[</span><span class="s1">&#39;acc_train&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">[:</span><span class="mi">300</span><span class="p">],</span><span class="n">predictions_train</span><span class="p">)</span>
        
    <span class="c1"># TODO: Compute accuracy on test set using accuracy_score()</span>
    <span class="n">results</span><span class="p">[</span><span class="s1">&#39;acc_test&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">predictions_test</span><span class="p">)</span>
    
    <span class="c1"># TODO: Compute F-score on the the first 300 training samples using fbeta_score()</span>
    <span class="n">results</span><span class="p">[</span><span class="s1">&#39;f_train&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_train</span><span class="p">[:</span><span class="mi">300</span><span class="p">],</span><span class="n">predictions_train</span><span class="p">,</span><span class="n">beta</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>
        
    <span class="c1"># TODO: Compute F-score on the test set which is y_test</span>
    <span class="n">results</span><span class="p">[</span><span class="s1">&#39;f_test&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">predictions_test</span><span class="p">,</span><span class="n">beta</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>
       
    <span class="c1"># Success</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{}</span><span class="s2"> trained on </span><span class="si">{}</span><span class="s2"> samples.&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">learner</span><span class="o">.</span><span class="vm">__class__</span><span class="o">.</span><span class="vm">__name__</span><span class="p">,</span> <span class="n">sample_size</span><span class="p">))</span>
        
    <span class="c1"># Return the results</span>
    <span class="k">return</span> <span class="n">results</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Implementation:-Initial-Model-Evaluation">Implementation: Initial Model Evaluation<a class="anchor-link" href="#Implementation:-Initial-Model-Evaluation">&#182;</a></h3><p>In the code cell, you will need to implement the following:</p>
<ul>
<li>Import the three supervised learning models you've discussed in the previous section.</li>
<li>Initialize the three models and store them in <code>'clf_A'</code>, <code>'clf_B'</code>, and <code>'clf_C'</code>.<ul>
<li>Use a <code>'random_state'</code> for each model you use, if provided.</li>
<li><strong>Note:</strong> Use the default settings for each model — you will tune one specific model in a later section.</li>
</ul>
</li>
<li>Calculate the number of records equal to 1%, 10%, and 100% of the training data.<ul>
<li>Store those values in <code>'samples_1'</code>, <code>'samples_10'</code>, and <code>'samples_100'</code> respectively.</li>
</ul>
</li>
</ul>
<p><strong>Note:</strong> Depending on which algorithms you chose, the following implementation may take some time to run!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># TODO: Import the three supervised learning models from sklearn</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="k">import</span> <span class="n">GaussianNB</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="k">import</span> <span class="n">svm</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
<span class="c1"># TODO: Initialize the three models</span>
<span class="n">clf_A</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">clf_B</span> <span class="o">=</span> <span class="n">svm</span><span class="o">.</span><span class="n">SVC</span><span class="p">()</span>
<span class="n">clf_C</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">()</span>

<span class="c1"># TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data</span>
<span class="c1"># HINT: samples_100 is the entire training set i.e. len(y_train)</span>
<span class="c1"># HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)</span>
<span class="c1"># HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)</span>
<span class="n">samples_100</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">y_train</span><span class="p">)</span>
<span class="n">samples_10</span> <span class="o">=</span> <span class="p">(</span><span class="nb">int</span><span class="p">)</span> <span class="p">(</span><span class="mf">0.1</span><span class="o">*</span><span class="n">samples_100</span><span class="p">)</span>
<span class="n">samples_1</span> <span class="o">=</span> <span class="p">(</span><span class="nb">int</span><span class="p">)</span> <span class="p">(</span><span class="mf">0.01</span><span class="o">*</span><span class="n">samples_100</span><span class="p">)</span>

<span class="c1"># Collect results on the learners</span>
<span class="n">results</span> <span class="o">=</span> <span class="p">{}</span>
<span class="k">for</span> <span class="n">clf</span> <span class="ow">in</span> <span class="p">[</span><span class="n">clf_A</span><span class="p">,</span> <span class="n">clf_B</span><span class="p">,</span> <span class="n">clf_C</span><span class="p">]:</span>
    <span class="n">clf_name</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="vm">__class__</span><span class="o">.</span><span class="vm">__name__</span>
    <span class="n">results</span><span class="p">[</span><span class="n">clf_name</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">samples</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">([</span><span class="n">samples_1</span><span class="p">,</span> <span class="n">samples_10</span><span class="p">,</span> <span class="n">samples_100</span><span class="p">]):</span>
        <span class="n">results</span><span class="p">[</span><span class="n">clf_name</span><span class="p">][</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> \
        <span class="n">train_predict</span><span class="p">(</span><span class="n">clf</span><span class="p">,</span> <span class="n">samples</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>

<span class="c1"># Run metrics visualization for the three supervised learning models chosen</span>
<span class="n">vs</span><span class="o">.</span><span class="n">evaluate</span><span class="p">(</span><span class="n">results</span><span class="p">,</span> <span class="n">accuracy</span><span class="p">,</span> <span class="n">fscore</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>GaussianNB trained on 361 samples.
GaussianNB trained on 3617 samples.
GaussianNB trained on 36177 samples.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>/opt/conda/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  &#39;precision&#39;, &#39;predicted&#39;, average, warn_for)
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>SVC trained on 361 samples.
SVC trained on 3617 samples.
SVC trained on 36177 samples.
RandomForestClassifier trained on 361 samples.
RandomForestClassifier trained on 3617 samples.
RandomForestClassifier trained on 36177 samples.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAxAAAAIuCAYAAAAv/u6UAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvpW3flQAAIABJREFUeJzs3XlcFVX/B/DPlx0UEcSFRUTEFVzQMDUr7UlLU3P5VS65L2Walj0uZaZZlj0ulbZYuZCZVqampqE95ZKPuZa4k6jghqiIisoi8P39cc7F4XovXJBVv+/X677gzpyZOTP3zJk52wwxM4QQQgghhBDCFnYlHQEhhBBCCCFE2SEFCCGEEEIIIYTNpAAhhBBCCCGEsJkUIIQQQgghhBA2kwKEEEIIIYQQwmZSgBBCCCGEEELYTAoQotgQ0QAiYsMnmYiiiGgkETkU8rZaEtFOIrqht9WkMNd/PyCiKfrYpRCRh4X5xt8zuIDrfyyfy8QSUUR+t1UQxZGGDMc4r88AIgrU/w8p7HgUNiJqQkQriOgUEaURUTwRbSKiUSUdt8Jg+t2KcXum335AHuFM52S+z8eSVJzntdl2I/TxOk1Ed9wPmZ2fhXKNMvxGgQVYloloSmHEQ4i7Vag3bULY6BkAZwBU0P/PBVAFwFuFuI0FAFIAdAZwE8A/hbju+80tAP8HdUyN+gFIBuBewPVOBjANwO/5WKYbgGsF3F5+FUcamg8g0vD9KQBv4vY5YnIcQLki2H6hI6JwAH8A2AlgHIDzAPwBtIb6/eaUXOwKjfnvJu5OcZ7X5m4C8AXQFsBvZvOex93lcULcs6QAIUrCPmaO0f9v1LVlr+AuCxBEZA+AAGQBqAtgGjPn5+bU2noJgCMzp9/tusqolQD6wlCAIKLqAB4FsBjAgKKOABE5M3MaM/9d1NvS27NDMaQhZj4DQ0GBiOrpf43niGlegQsQxZyGXwZwBUB7Zk4zTF9iqZa3tDClMVvCmv9u4raCpLXiOq+tSAJwFCqPyy5AEFFrAEFQeVz/komaEKVXqc3MxX1lNwB3IqpimkBEQ3X3plQiukREC4jIy7iQbs6dRkQTiOgkgHSom5dMqLQ9SYeJNSzzvNl6vyEiH7P1xhLREiIaRERH9XqfMnQjeJGI3iei87ob1hIiciOiYCLaQETXiSiGiPqbrTdYb++k7hZ0gog+JyJPs3ARRHSGiMKI6A8iuklEx4joRfMDR0Q19TrP664iJ4joY7MwjxLRbzquN3QcQ/Px+ywG8AgR1TBM6wvgFICtlhYgou5EtEPH/QoRLSeiAMN8U/ePiYYuAlPM9r8lEW0nohQA/9Hz7ujqkNcxIKJwIvqViBJ1fE4Q0WfWdpZUN5EiSUPWtplP9kQ0lVS3oCtEtJaI/G3dvk6rH+h0mK7/TjS/uScib50+z+rjepSIhtkQPy8ASZZuxpk5y7D+NvrYtjHb7h1dPAz7M1SfW6lE9BcRtTXfhi3pnYg2E9E2IupMRH8TURqAl4joEBGtsLDOB3Wcuurvd3RhIqLRRHREn9tJRLSHiLqZhcn1vNBh3IjoM51erxPRGqgWnEJj4zFqT0TrdTq7SUQHieg1UhU1xnB55Zcv2JheIwzfTWmgBRF9S0TXiOgcEc0hIhezZYN0PG8S0QUimkVEw8zTUB4WA+hBRG6Gaf2gWtJiLRw/RyJ6V8c7Xf99l4gcLcRtnY7bRVL5krOlCJAN1zwLy9QholV6v1NJdRlcToXcJVgIi5hZPvIplg9UTTUDCDabvhxABgA3/X06VLeZWQDaAxgI4CxUlwh7w3Ksp/8BoAeAJwFUBfCQnjcfQAsAYTr8MD39OwAdAQwBcAGqa0p5w3pj9XoPAugF4F8AagEI1MvHAfgawBMAXtVxXQzgAIBRANpB1dpnAQgxrPcRAO8DeFr/P0Bv+0+z4xEB1Zx/BMALen1L9bbbGsLVBHBRx+cFAI9B1ZR9awjzlD62q/V2nwawHarWrXoev9cUvU1HACcAvGGYdwTAO5Z+UwAv6mkL9XF+Toc/CcBdh2mhwyzS/7cA4G/Y/2S9Xy8DaAPgQcNvE2HrMQBQHsBlqO4mnfW6BgD4Mpf9rowiSkMFPUf0vEA9L1anhw56Xy8B2GIW1uL2oVqd/wCQCNXq9y8AEwGkAphlWL4CgGioQuJQAI8DmAFVsHo5j314S8dzHoDmAByshGujw7WxcgwCzfbntE5HzwHoCuBPHe+6+U3vADbr3+0kgEE6Lo0ATACQBsDTLE5z9TFzMp4bhvl99HbfguoK01Gva3B+zgsd7huom/CJUPnfDP07MIABBU0/BThGLwJ4TaeztgDGQp2X021Ma4HIX3qNsLAfxwBMhUp/k6DS39uGcE5Q3fvO6mU6QuW9cTBLQ1aORQRUS1I5ANcB9NbTnfXxGIzb+aCDYbml+hhO1b/RZKjrwFILcTsHdQ17CsAaqHRsnr7zc82bYvj+D4BdUNe/RwH0BrAEOp3KRz5F+SnxCMjn/vkYLgp1oW5kPKFu+jIB/KTDBOrvb5kta7qh62qYxjpzdjUL62Aho7UHkABgk1nY1jrsKMO0WKh+sdXMwgbqsL+bTV+ppz9vmOapLzCTczkeDobthxmmR+DOwoIz1IX3S8O0xfqi55vLNmIA/GY2rYJe10d5/F7ZF059oTyipzfX02vD7IYF6ob9KoCFFo5dOoBXzH6/dy1s17T/T1uYF4ucNxq5HgMAD+h1NcpnWi2SNJSPcyS3AoT5zde/9XTfvLYP1XLEAB4xmz5R/z5V9PdJUDfntc3CfaXTjsVCgQ7jCmCV3g7reGyEKogYb4baIH8FiHQAAYZp7lCFw2/ym96hChBZAJqYha0Olf+8YJjmCFVI/cz83DB8/wTAX7kcE5vOC6i8MRPABLNwn6PwChD5zhOguoY66HSSBMDOhrSW3/QaYWE/3jZb9mcA/xi+mwr0zc3iGmWehqzsVwSAM/r/xQAi9f/P6n2qALMCBIBQmOUNevqbMOQ1UOmdAbQwhLEDcMgYN+T/mjdF/++tv3fJbR/lI5+i+kgXJlESjkLVtlwG8BmAb6FqAQFV224H4FsicjB9oGpirkHV3BtFMnOKDdusCzVQ+1vjRGbeBlVb9ahZ+B3MfN7Kun6xsD8AsMGw3iSoGs7qpmlE5EREb+iuIClQx+APQ/yMbjLzJsP60qBq44zdHdoD+JmZz1mKJBHVhqoJND+WN6Fqb82PZW4WA6hHaoBsP6jjc8xCuJZQF13zbZ6BOk62bjMD6mYhL7keA6hjdgXAF6S6HlW3Es4WhZmG7sY6s+8H9N8As+mWtv8kVFy3m/0+G6FulFsYwu0EcNIs3AYAlQA0sBY5Zk5h5m4AQqBqrX+BKsh9CWA9EVE+9tV8f04ZtpMMdSxaAgVK77HMvM8s7qcBbIEqaJk8CXWztjiXuO0G0ISI5hLR42ZdYQDbz4sHofK/H8yW/y6XbdssP8eIiHyI6AsiioMq5NwC8C6AilDngVFuad3W9GrrssblWgA4xcy7TBOYmQHc0Q3NBosBPE5E1aDyuNXMbGlgt+kYLTGbbvpuygdaAjjNzDsMccvCnb9tfq95JolQLcPTdfen2nnuoRCFSPrJiZLQDerCmQwgjplTDfNMF6aYO5ZSKpl9j7dxm6a+pJbCnzfMt2W9SWbf03OZbuyv+z5Ul5ypUF0GkqH6Nq80C2dpXYDqWmEMVwm5D+Q0HcsFuPMJSoDqFmETZo4hoj+hmvT/D6qGOrdt/tfKfEv7ZckFZs60IVyux4CZr5LqJz8JqrDqTkSHoFqG8nuTUZhp6G5cNvtuGmtgnoYsbb8KgBpQN4OWVDKEC7YhnFXMfBjAYQDQ/da/gnqqzVOwrXBoLsHKND/9f37Tu7XfZzGARURUk5lPQhUmYow3glaWcYE6P14CcIuI1gMYw8yxsP28MI2lMd9XS/teEDYdI1LjYdZAPZ1oClQhJwWq69hE2JbWTGxNr7YuaxxD4ANVUWOuIMfrd6j9eBWqe2oXK+Gs5QPnzeb7WImH+bT8XvMAqIISEbWD+n3eB1CJ1FjAGcz8uZV1CVFopAAhSsJBNnvCjEGi/tselm82E82+s43bNF2IqlmYVw3AngKuNz96AljMzO+aJhBR+btY3yXcvnmyxHSsXoflG5f8PpFnMYBPoVoHvs9jmwOgmurNJdu4LVuPf17HALqWuYeu1XsA6nj8QESNmfmgjdsBSkcayg9L20+E6nP/rJVlYg3hLgAYbSVcdL4iwpxKRDOgChANoAoQpooDJ7Pg1gonVa1MO6v/z296t/b7rIBK58/rQa+doW7QrNK13l9AtXR5QuVfs6DOkwdh+3lhuimtClW7DMP3wmDrMaoFda70ZebsmnYi6mxlvSWV1uNhuTUs38eLmbOI6FuoVrMLUK1ylhjzgeOG6aZ8wXSM46Fa4fKKW36vecY4nwDQT7fqNQYwEsBnRBTLzOYt5UIUKilAiNLmV6i+yQHM/GshrjcaquanJ3I+jrQVVI3srELcljVuuLNGd+BdrG8jgO5E5MPMlmoAo6FuCEOYefpdbMfke6iauf3MbF4zaGJqWQlm5q/zWF86VH/5u5HXMcjGzBkAdhDRJKjaxfpQAz9tVRrS0N2KhBpweZ2Zj+YR7mWo7iGWanitIiJ/Vo85NWd6RK3pd4rTf0OR82ato5VVtyCi6rqbEYjIHao1w9TNpVDSOzMnE9FqqJaHc1A15d/kY/kkAN8T0YNQY7wA28+LnVD537NQA2tNetq+B7my9RiZumBl51f6CUN9CikehWUHgIFE1NzUjUnfTPco4PoWQqXTX3NpAd2i//aEeo+NienYmJ5M96eOWwtT65Vu2TEvvN/1NU8XYPcR0RioVrBQ3NnVVohCJQUIUaow83Ei+gDAJ0RUFyqzToUaS9AOwHzj2IB8rDeTiN6CqiFcAtVf1Q/qAnAM6mlARS0SQH8iOgDVXN0dQKu7WN9kqBuo7UT0nl6nH4Anmfl53cQ9AsBqInKC6nt7CaoGrBXUzeFsWzemb4y65RHmGhGNBfApEVWGuohd1fF6FMBmZl6qgx+GetxjJFTN27lcxjJYk+sxIKJOUAMtf4KqeS8H9aSsZKgLvM1KSRq6W99CFVp/I6JZUINNnaBqnLtADdi8CeBDqKcE/UFEH0LdeJaDurl6mJmfzmUb84ioKtRN90GowefhUC+VOw41wBrMHE9EWwC8TkSXoGp9n9dxsSQB6r0xU6C6sozXcXpHr68w0/tiqCcKvQ1gm+7KZBURfYnbaeoCgDpQBZCNOm42nRfMHE1ESwFM1Tebu6HyPWuFKmueJCLzMQlXmflXG4/REagC3jQiyoQqSLyazzgUhwiodLCSiCZCDXYfAvUQC0DdmNuMmf+B6qaVW5hDRLQMwBTdqrkdarzDJADLmHm/Dvo11JO4VhLRG1Dp4kWosTDG9RXomkdEjQB8DFWxEwN1ng2AaiG+63fXCJEXKUCIUoeZ3yCiIwBG6A9DPfruN6gbtYKu90siugnVRL0a6uk96wGMY+brdx3xvL0M9YQQU63VeqiblF1Wl8gFM8fqWs53obpYuEN151htCLOeiB6B6rc8H6rG/zxUzZ21bkh3hZm/IKLTUMe5N9Tg3LNQNXPGQasjod5KvBaqX/PbUP1587OtvI7BMai+25Og+iQnQ9+UWaklz2t7JZ2G7goz3yKiJ6BubIZBPQb3BtSN/TroLix67EgrqMeSjoe60b0CVZDIa+zIXKjffQRUH3onqHEqSwC8Y3acnod6wtAcqJumhVC/5VcW1rsF6ulJ70GNHToMoIO+6TPtX2Gl91/1cn5QY5by8j+ogllfAB5QLRdLoAq4prjZel68AJWu/g117H7X4bflI/5zLUw7BCDUlmPEzOmk3nnxCVRh6jLUb3MKln+bEqHj2R5qf+dBHbelUC0506EKaUWhP1QXs0FQT186B+ADqDzMGLd2UMfwM6jzbCnUeTbPbD8Kcs07D/V7jIE6H1KhBpl3Yua9hbKXQuSCVMuXEEIIUTqRepHfNmZ+vqTjIko/IvoZQH1mttaaJYS4S9ICIYQQQogySff7vw5VU+8O4Bmobo3DSzJeQtzrpAAhhBBCiLIqDWp8RgDUOIBoAEOY2dJjaoUQhUS6MAkhhBBCCCFsJm+iFkIIIYQQQthMChBCCCGEEEIIm0kBQgghhBBCCGEzKUAIIYQQQgghbCYFCCGEEEIIIYTNpAAhhBBCCCGEsJkUIIQQQgghhBA2kwKEEEIIIYQQwmZSgBBCCCGEEELYTAoQQgghhBBCCJtJAUIIIYQQQghhMylACCGEEEIIIWwmBQghhBBCCCGEzaQAIYQQQgghhLCZFCCEEEIIIYQQNpMChBBCCCGEEMJmUoAQQgghhBBC2EwKEEIIIYQQQgibSQFCCCGEEEIIYTOHko6AEHfjr7/+esLBwWEyM1eDFIiFEEKIwpJFROczMjLebtq06YaSjowoXYiZSzoOQhTIX3/99YSzs/MngYGB6a6urql2dnaSmIUQQohCkJWVRSkpKS6xsbFOaWlpI6UQIYykxlaUWQ4ODpMDAwPTy5UrlyKFByGEEKLw2NnZcbly5VICAwPTHRwcJpd0fETpIgUIUWYxczVXV9fUko6HEEIIca9ydXVN1d2EhcgmBQhRltlJy4MQQghRdPR1Vu4XRQ6SIIQQQgghhBA2kwKEECJPY8aM8Q0ICAgt6XgIIQquefPmdZ977rkaJR2Psuznn392J6Jmx48fdyyO7UVHRzsRUbMNGzaUN02LiYlxbNmyZR1XV9cwImoGAH5+fg3HjRvnUxxxEgKQx7iKe8x33t6N0xITizVdO1eqlNHz0qWogiybkJBgP3Xq1GqRkZEVz5075+To6Mi+vr7p7dq1uzp69OgLwcHBtwo7vgUxefLk82PHjr1Q2OsdM2aM74cffujToUOHpPXr158wznNwcGg2e/bs2FGjRiUC6gJ57tw5JwAgIlSsWDEjLCzs+syZM8+GhYWVyFgY7+++a5yYllas6a2Ss3PGpZ49bU5v169fp4kTJ/r89NNPXgkJCU4uLi5Z/v7+aT179kx88803LwwcOLD6unXrPM+ePbvf0fHOe6JatWqFhIaG3ly9evVJADh//rz9lClTfCIjIyvGx8c7lStXLjMoKCh1wIABl1544YVES+soat7feTdOTCve876Sc6WMSz3zd9736NEjcOXKlZUAwM7ODt7e3rdatWp1bfbs2Wdr1qxZKs71whAdHe1Ur169hubTAwMDU0+ePHmoJOJkZJ63GH322WdeCxcurBwdHe2akZFB/v7+6f/617+ujh8/PqEkfqNatWqlx8XFRVWtWjXTNG3y5Mk+iYmJDrt27Trs4eGRCQC7d+8+Ur58+azijp+4f0kLhLinFHfh4W62GRMT4xgWFtZgzZo1nmPGjInfvHnz0W3bth2ZOnXqmcTERPtp06aVmkFrHh4eWT4+PhlFsW5nZ2eOjIz0/O9//1sur7DDhw8/HxcXF3Xy5Mn9K1asOJacnOzQpUuX4KKIly2Ku/BQkG3279+/xvLlyyu9++67Z/bt23fwl19+iR42bNiFK1eu2APAiBEjLl68eNHx+++/r2i+7MaNG8udOHHC5cUXX7wIAMePH3ds2rRpg3Xr1lUcP378ue3btx/esmXL0f79+1+aM2dO1d27d7sWzl7mT3EXHu5mm82aNbseFxcXFRMTsz8iIuLEoUOH3Lp3716rsONXGixZsiQmLi4uyvTZvn17dEHXlZWVhbS0NCrM+Jl79tlna7zyyiuBrVq1Sl65cuWxqKioQ7NmzTqVkJDgMG3atKpFuW1rHBwcEBAQkOHs7Jw93u/kyZMuTZo0udGwYcO0gICADADw9fXNqFChwl0VIFJTU4v0+Ip7ixQghCghw4YNq3Hr1i2Kioo6PGLEiMsPPvhgSqNGjdKeffbZa0uXLj21YMGC0wCwatWqCs2bN6/r4eHRxN3dvUl4eHjdTZs2uRnXRUTNPvvsMy/jtFatWtXp0aNHoOn7kiVLKtavX7+Bq6trmLu7e5OGDRvW/9///ucKAGlpaTRkyBD/qlWrNnJycmpauXLlRp06dQoyLWveheno0aNO7du3r1WlSpVGrq6uYXXq1Gnw6aef5ti+qbvE2LFjfby9vRt7eHg06datW+DVq1dz5DtVqlRJf+KJJ5LGjRtXPa9jVr58+ayAgICMGjVq3Grbtu3NV1555fyZM2ecL168aG/TQb8Pbdy4seLIkSPP9+3b90q9evXSW7ZsmTJq1KjEmTNnxgPAAw88kNq0adPrCxYs8DZf9osvvqhcs2bN1A4dOlwHgKFDh9ZIT0+327dv35Hhw4dfbtasWWrDhg3TXn755cQDBw4cCQ0NTSvu/StrnJycOCAgIKNmzZq3OnTocL1///6X9u3bV+7y5ct2ADBv3jyvRo0a1XN3d2/i6enZuE2bNsH79+93Ni1v6tIyf/58z8ceeyzY1dU1zN/fv+GcOXMqGbfzzz//OD388MO1XVxcmvr4+DScNm1aFfO4JCUl2fXu3buGp6dnY2dn56ahoaH1V65cWcF8W/PmzfNq3bp1bVdX17CaNWuGrFu3rvzJkycdH3300WBXV9ewWrVqhURGRpY3X7+3t3dmQEBAhuljrISIiopybtOmTbCbm1uYm5tb2GOPPRZ88ODB7P2cM2dOJQcHh2Zr1651r1+/fgNnZ+emq1evdgdUnti0adN6Li4uTatUqdLo//7v/wLPnz+fnQfs2bPHpXXr1rXd3d2buLq6hgUFBYWY8ic/P7+GmZmZGD16dCARNTN1AYqIiKi4fPly708++eTknDlzzrVr1+5GnTp10rt06ZK8atWq2Pfffz/e0u+ZlZWFnj171qhevXqoi4tLU39//4YjR470S0lJyb4ZP378uOMTTzxRy9PTs7EpzKRJk7ILJLnlzeZdmIio2Z9//um+fPlybyJqZsrjzbsw3bp1C2PGjPH18/Nr6Ozs3DQ4ODhkxowZOc5xImr27rvvVuncuXNNd3f3Jj169KhpaR+FsEQKEEKUgISEBPstW7Z4DB48+IKXl5fFWiM7O3V6Jicn2w0bNuzC1q1bj2zatOloUFBQateuXesYL5h5OXXqlMPAgQODevTokfj3338f2rJly9ERI0YkmLqbvP/++1XWrl3rtWDBgpOHDh06+OOPP8Y0b978urX1Xbt2zb5NmzbXVq9efWzPnj2H+/fvf2n06NE1165d624Mt379es/Lly87/Prrr9GLFi068fvvv1d866237mhZmTlz5tmDBw+6RURE3FELbs2lS5fsly1b5hUUFJRauXLlzLyXuD9Vrlz51q+//uqRkJBgNb0MHDjw0h9//OFh7Nd9+fJlu/Xr13v279//InA7zQ4aNOhCpUqV7jjezs7OfLc1oPeb2NhYx59++snT3t4eDg6qQSMtLY0mTJgQv2PHjiM///zzP3Z2dujcuXNt89rhKVOm+Pfu3Ttx9+7dh7t27Xp5zJgxgaaCRlZWFrp27VorKSnJYf369dErVqyIWbduncfhw4dzVDz07t07cMuWLRXmz59/8s8//zwcHh5+/bnnngv++++/XYzhpk2b5vvCCy9c3Llz5+Hg4ODUgQMHBvXp06fm4MGDL+3cufNw7dq1UwYMGBBkawvB9evXqUOHDnXS0tLsNmzYEL1hw4boGzdu2HXs2DHHfmZlZeH111/3/89//nM6KirqYOvWrW+uWbPGvXfv3rV69Ohxeffu3YeWL18ec/r0aadOnToFZ2VlmfYryNPTM2Pz5s1H9+7de2j69Omnvby8srv62NvbY+rUqadNLSMAsGTJkkoBAQFpw4YNS7IUZ2t5DDOjcuXKGV9//fXJqKiogx988MGp77//vtIbb7yRfTM/dOjQGsnJyfbr1q37Jyoq6uDnn38e6+/vfwvIO282FxcXF9WkSZMbnTt3vhwXFxf15ZdfnrYUrmfPnoE///xzxblz58ZFRUUdHD9+/LmpU6f6f/jhhzkKETNnzvRt2bLl9R07dhyZPn36Weu/mhA5yRgIIUrA4cOHnbOystCgQYMcfffDwsLqRUdHuwKAr69vekxMzKF+/fpdMYZZunRpnKenp+eqVas8hg8fftmW7Z0+fdoxIyOD+vbtm1S3bt10AGjatGn2tuPi4pxq1qyZ2rFjx2Q7OzvUrl07/dFHH71pbX3NmzdPad68eYrpe0hIyIXff//d/dtvv/Xq3Llzsmm6j49PuqklJSwsLHXFihWXt2zZUgHAOeP6QkJC0vr27Xtx8uTJ/r169bpqbK43+uijj3w++eSTasyM1NRUOz8/v/R169b9Y8sxuF/NmzcvdsCAAUG+vr5NatWqldKsWbMbHTt2vNqnT58rpkLqoEGDLk+cOLH6559/7m1qmfjqq68qZWVl0QsvvJAI3E6zISEhKblsTuRh165d7m5ubmGmNAwAQ4cOTTAVvkaPHp2jX/6yZctOVqtWrcnWrVvd2rdvf8M0fciQIReGDBmSBAAfffTR2UWLFlXZsGGDe6NGjdLWrFnjfuTIEbeoqKiDjRo1SgOAH3/88WRQUFAj0/IHDx50joyM9Pzuu+9ievTocQ0AFi1adHrnzp3l33vvvWrLly+PNYUdOnTohb59+14BgDfffDO+TZs29YcPH55gypsmTZoU37p16wb79+93Dg8Pz85XunbtWpvodpnigw8+OPXyyy8nfvXVV5WSkpIc9u7de8TUKvHjjz+eqF27dqP58+d7jRw5MhFQN+czZ848/eSTT2ZXZrz77ru+AwcOvDBx4sTsMVlLliyJrVOnTsMdO3a4tmrVKiU+Pt5p5MiRCc2aNUsFgAYNGqSbwvr6+mYAgIeHR6ap+w+gugXVqlUr32Op7O3tMXfu3Owb77p166bHxMScX7hwYZUPP/zwHACcPXvW6amnnrrSqlWrFFMYU/i88mZzAQEBGY6Ojuzq6ppljL/R0aNHnVatWlVp7969h0zjw+rVq5ceHR3t8sUXX1R59dVXL5nCtm/fPumNN964mN/9FkJaIIQoAcxssaZu+fLlx3ft2nW4d+/eF1NSUuzVtCEIAAAgAElEQVQAdTHo2rVrzYCAgNDy5cuHubu7h12/ft0+Li7OydbtPfjggymtW7e+FhYWFtKuXbta77zzTpWYmJjsKq6hQ4deio6Odq1Ro0Zo7969AyIiIirm1h82OTnZ7qWXXvILDg4O8fDwaOLm5ha2ZcsWj9OnT+eIU0hISI5CiK+v761Lly5ZrFp77733ziUlJTnMmDGjsrXt9uvX78KuXbsO7969+3BkZGR0cHBwSpcuXWonJSVJXmZF+/btb8TFxR2IjIw82qtXr8QLFy44DBw4sNbjjz+eXWPr5ubG3bt3T1y6dKl3ZqaqaP3666+9n3zyyaRq1aplArfTrPGGUORfo0aNbuzatevw1q1bj4wePTq+SZMmNz766KPsG9Dt27e7tmvXrpafn1/DcuXKhdWsWbMRAJw4ccLZuJ6mTZtmn1sODg7w8vK6lZCQ4AgABw8edK1YsWKGqfAAqBvnwMDA7BvTqKgoFwB48sknk43rbdGixfXo6OgcLRBhYWHZhUY/P79bANCkSZPsaf7+/hkAEB8fn+PcnjNnTuyuXbsOmz59+vRJAoBDhw651KpVK9XYpal69eoZgYGBqYcOHcqx7UceeeSG8fuBAwfcFixYUNXU9cnNzS2scePGIQBw9OhRFwB48cUXE8aMGRPYvHnzumPGjPHdtm1bjpYXS5gZRFSg9wrNmjXLu1GjRvUqVarU2M3NLey9997zNz3wAQBeeumlhLlz51Zr1KhRveHDh/v98ssv2d298sqbC2L79u3lmBkPPfRQfeNxmjt3rk9cXFyOdBQeHn7D2nqEyI1cdIUoASEhIal2dnYwv1gGBwffCg0NTTM1twNAp06dap89e9bpww8/PLVly5Yju3btOuzl5ZWRnp6eff4SEZhzXvtu3bqVfafn4OCALVu2HFu3bl10s2bNbqxevdozNDS04bJlyzwAoFWrVimxsbEHpk2bdsbJyYnHjx8fEBIS0sDUL9vcSy+95L9ixYpKEyZMOBcZGRm9a9euw48++ujVW7du5Qjv5OSUI1KW4mlStWrVzFdeeSV+1qxZPomJiRa723h5eWWGhoamhYaGpj3xxBPXFy9eHHvq1CnnRYsWeVkKLxRHR0e0a9fuxttvv53w22+/HZ8zZ87JTZs2eRhvZEaMGHExPj7eacWKFRX++OMPtyNHjriZBk8Dt9PswYMHS2Sg9L3CxcUlKzQ0NC08PDz1o48+Ole9evW0QYMGBQCqYN6pU6c6RIQvvvgi9o8//jiybdu2I0SE9PT0HCU381Y6IkJWVhYB2TfDBYqfpWWN57FpnqOjo3EaA6rLkVFAQMAt0/mq87Vcu7iZb9ve3h5ubm5sFoZeeuml88aCya5duw4fOHDgYI8ePa4CwIwZM+L3799/oHv37pcPHz7s0rZt23qjRo3yzW3bQUFBqTExMflO2wsXLvScMGFCQPfu3ZN++umnYzt37jz86quvnsvIyMjekdGjRyfGxMQcGDx48MXz5887du/evfbTTz9dE8g7by4IUyXApk2bjhqP0d69ew/t2bPnsDFsuXLlpNuhKBApQAhRAqpWrZr5yCOPXF2wYEFVazfLgHpk5vHjx13Gjh0b36NHj2vNmjVLdXV1zbp8+XKO7odeXl4ZxhqvlJQUiomJyVE4sbOzQ9u2bW9Onz79/J49e6LDw8OTIyIisvvDenh4ZPXr1+9KRETE6d27dx8+ceKES2RkZI4xDSY7d+4s371798QhQ4YktWzZMqV+/fppJ0+edLEUNj9ef/31C25ublkTJ0606Xnmpn7jptYaYZuGDRumAoCpxhq4PZh6/vz5lefNm+dtHDwN3E6zCxcurGIpzaalpdG1a9fkd8inadOmnfvxxx+9t27d6rZv3z6XpKQkh+nTp5/t1KlTctOmTVMTExPtrRW6rWnYsGFKUlKSw4EDB7Jrm+Pj4x2Mtc9NmjRJBQDzc3znzp3l69atW6Td1EJCQlKPHz/uEh8fn52PnT592iEuLs4lry5yISEhN44cOeJiLJiYPh4eHtk3ww0aNEifMGHCxcjIyBNjx449t3jx4uxB5I6OjlmZmZk5Skl9+vS5fOrUKecvv/zS09J2rT2oYcuWLeXr169/c8qUKQkPP/zwzYYNG6aZ1/IDQI0aNW6NHj06cdWqVbEff/xx7Jo1a7xMFTR55c351bJly5sAcOLECSfzYxQSEiIPOhCFQsZACFFCvvzyy1MPP/xwvcaNGzeYMGHCufDw8Jvu7u6ZBw8edNmwYYOHnZ0dV65cOdPT0zNj/vz5levVq5d24cIFh/Hjx/s7OzvnqDV66KGHrkVERFRu27ZtsoeHR+bUqVN9jDVgv/76a7mNGzdW6NChw7Xq1avfOnz4sHN0dLRrr169LgHApEmTqvr6+t4KDw+/Wb58+ayIiAgve3t7hISEWOyLGxQUlBoZGVlx06ZNSRUqVMj6z3/+U/XixYuO3t7ed/WoV1dXV37rrbfOjho1KtC8NhMArl+/bnfq1CkHADh79qzj22+/7ePi4pLVuXPnq3ez3XtZeHh43WeeeeZyixYtblSrVi3jyJEjzpMmTfJzd3fP7NChQ47uKwMHDrz0yiuv1HBxceGxY8feMaDSlGbDwsLqv/HGG+fCw8NvOjs789atW8t9/PHH1RYtWnTS1M9b2KZhw4Zpbdu2vfLGG2/4LV++/KSTkxPPnj27yuuvv54QExPjNHHiRP/8tiZ06dIluW7duil9+vSp+fHHH59ydnbmcePG+dvb374HDgkJSevQoUPSmDFjAhwcHOKCgoLS58yZU/nYsWOu33777cnC3k+joUOHJs6YMcOne/fuQTNmzDjDzPj3v//tX6VKlfTBgwdbHMRsMmXKlHPdunWrPWTIEP9BgwYlenh4ZB05csT5hx9+8Fy4cOGpzMxMGjlypP8zzzyTVKdOnbTExET7//73vx61atUydsNK37x5s3u3bt2uOjs7s4+PT8bAgQOT1q5dmzhy5Miahw4dcu3cufPVGjVq3Prnn3+cFi1a5F2xYsWM+fPnnzGPT926dVN/+OEH7yVLllQMCwtLWblypUdkZGSOh0H069cv4KmnnroaGhqampKSQj/99JNntWrV0itWrJiVV95cEKGhoWnPPPPMpVGjRtW4fPnymUcfffRGcnKy3c6dO90uXrzoOG3atPMFXbcQJlKAEKKE1K5dO/3vv/8+PHXq1KqzZ8+udu7cOWcA8PPzS2vTps21cePGJdjb2+Obb745PmbMmIDw8PAQHx+f9ClTppx56623/I3rmjt37ukBAwYEdu3atU758uUzX3311fjExMTs2mVPT8/MXbt2lVu0aFGVa9eu2Xt7e9/q1q3b5Q8++CAeACpUqJD5ySefVI2Li3PJyspCUFBQakRExPHGjRtbrK365JNPTg8YMCCwY8eOdcuXL5/5/PPPX+rQoUNSbGzsXbdCDB069PKnn35a5cCBA3e8F+Lzzz+v9vnnn1cD1CDI+vXr31y5cuUxY19vkVO7du2ufv/9917Tp0/3vXHjhr2Xl9et5s2bX1+0aFGs+bs9TIOpU1NT7UyDp41q166d/tdffx2ePHlytenTp/uaXiRXq1at1NGjR58PDw+XwkMBjBs37nz79u3r7d2713XevHknp0yZ4rd8+XLvoKCg1JkzZ556+umn6+ZnfXZ2dli9enXMoEGDajz55JP1KlasmDFy5Mjzxm6PgBp8PHLkyOpDhgypeePGDfs6deqkfP/99zFF/WLG8uXL8y+//PLPqFGjqrdv374uADz44IPJ69evP+bi4pJrc0vnzp2Tf/7553+mTp3q265du8pZWVnw8fFJb9OmzTVnZ2e+desWrly5Yj98+PDAS5cuOZYrVy6zZcuWyR9//HH204qmT59+ety4cdVr167dMCMjg5h5LwCsXLkydu7cuckRERHeCxYsqJqZmQl/f//0du3aXRk/frzFF2m+9tprlw4ePOg2YsSIwMzMTGrbtu2VsWPHnps4cWKAKQwzY/z48dXPnz/v5OLiktWkSZPra9euPWZnZ5dn3lxQS5cujZsyZUrVmTNn+rzyyivO5cuXzwwODk4dPnx4ob8QVNyfKL9No0KUFlFRUbGNGzfOUUtT1t5ELcq2svAm6vtBWXkTtRBlVVRUlHfjxo0DSzoeovSQFghxT5EbeVGc5Ea+dJAbeSGEKF4y4E0IIYQQQghhMylACCGEEEIIIWwmBQghhBBCCCGEzaQAIYQQQgghhLCZFCBEWZZlevOqEEIIIQqfvs7KG6tFDlKAEGUWEZ1PSUm56/cOCCGEEMKylJQUFyKSl8+JHKQAIcqsjIyMt2NjY51u3LjhKi0RQgghROHJysqiGzduuMbGxjplZGS8XdLxEaWLvEjuPkREgQBOAnBk5ow8wg4AMISZWxdDvNoA+JSZQ2wNu3fv3jEODg6TmbkapEAshLDgwoULfh4eHonOzs55vmE5P2GLS1pamsuVK1cqVa1a9WxJx0XcV7KI6HxGRsbbTZs23VDcGyeijQC+ZuZvCzNscSGiYADHmPmerOCUAkQpR0SxAHwB+DLzJcP0fQAaA6jJzLH5XGcg7rIAQUQPA/jF9BWAG4AbhiANmPlUfuIlRGEgos1Q50Y1Zk4r4egUCSJ6GsDbAIIApAOIAjA4v3lBaUREhwDU0F9dAdwCYMqn3mPm90okYneJiJwBfADgGQAVAFwCsIKZ/23Dso8DmM/MgYUcpzMAnmfmzYW53vuNvk5XBZBpmFyHmc+VTIyKHxH9AuBh/dUZAEPlTQCwhJlfLJGI3SUiIgATAQwB4A3gCoAtzNzHhmWLpABBRNug8oOIwlxvfsmbqMuGkwB6AZgLAETUEOrCWmKY+Q8A5XV8AqHiWNFagYSI7PRyMhBLFBmdFh8GcBVAFwDLi3HbDnkVyAtpO8EAFgPoDuB3qPOwPQpxkKO+aFJJnK/GFkhdGFzCzPOthS+u414I3gTQCEAzAAkAAgE8VJIREoWqMzP/t6QjQUT2zJyZd8jCxcwdDHGIAHCGmd+0Fr4MnbeDAPQE8BgznyAiHwCdSjhOpYJ0+SgbvgHQz/C9P9QNRDYi8iCixUR0kYjiiOhN0007EdkT0UwiukREJwA8ZWHZBUQUT0RniehdIrK/20gT0TYieoeI/oRqnQggoiFEdISIkonoOBENMYR/XNfkmL6fIaIxRHSAiK4S0TJdi5evsHr+60R0Xu/fUCJifbMp7i39AOwAEAF1nmQjIlcimqXPj6s6fbrqea2JaDsRXSGi07rlDUS02SyNDtC1P6bvTEQjiOgYgGN62sd6HdeIaK9urTOFtyeiN3TaT9bzqxPRp0Q0yyy+a4noFQv72ATASWb+jZVkZl5havGztg09rxUR7db7v5uIWhm2t5mIphHR/wDcBBCUn7yBiJyJ6CMiOqc/HxnO1zb6HH2NiC7o9Q3M/ae0TOchW4loDhFdBvAmEdUmok1ElKjzuW+IyMOwzBlS3R6h92EZES3Rx+cgETUtYNgHiGifnvcdES0noilWoh4OYCUzn9e/20lmXqLX42CeJ+lt5lgXEb2l9/EkEfU0TO9Et/PVM0T0qmFeFyKK0ml7GxGF6unLoFq3fyGi60Q0Jl8/hCgQnYec0L/VSSLqY5g31PA7HjalNSKqr8/PK0R0iIi6GJaJIKLPiWg9Ed0A0FafizOJ6BQRJRDRPNJ5nYX42JG6X4jT5+Zi07lDRIE6XfbX67pERBMLuN+PE1GszpvOA/iKiCrpeF8koiRSeZ6fYZltdDsvHkJEW4joQ30cThBR+wKGraXDJxPRRn38IqxEPRxAJDOfAABmjmfmrwzrys4v9Pd3zdelf1dTvmg8N1sQ0V+krhUJRDTDMO8hItqh47+PiB7R0z8A0BLAPH3efmTrb1DomFk+pfgDIBbA4wCiAdQHYA/gNFQTPwMI1OEWA1gNwB2qZusfqC4NAPAigKMAqgPwArBJL+ug5/8E4AsA5QBUAbALwAt63gAA2/KIY6BxfYbp23T86wNwhGrx6gzV7YIAPAYgBUAjHf5xALGG5c9A3QxWA1BJ79OQAoTtBOCcjkc5AMuMx04+984HQAyAl6BqeW8BqGqY9ymAzQD89HnUCqqpPQBAMlQrn6NOP030MptN6Uh/z3E+6HT0qz6vXPW05/U6HAC8BuA8ABc9byyAAwDq6nOgsQ7bXKdROx3OG+omvqqFfQwCkArgQwBtAZQ3m29tG14AkgD01XHrpb9XMuzrKQAher4jcskbLMRrqj4HqwCoDGA7gHf0vDZQ3ZCm6vV21PvnmcfvmeP462lD9LqG69/RFUAdAP8C4KS3/z8AMw3LnAHQRv//LlS+84RefobZb2pTWJ12zgAYqffpGag0N8XKvkwBEKfjHQrdhVjPc4BZngRgiWldUPldht6+M1TeeRNAsJ5/EUAr/b8XgKb6/3Co1o5wHf9BAI4DcDLfV/ncVb4TC+BxG8KVA3ANQF393QdAiP7/GQBn9W9FAIKhrvOOUPnaGzp9PwaVX5nWEQHV4voQVKWwC4CPAKzRacEdwFoA71uJ0yC9/iCo1syVAL7R8wJ1uvxKn2eNAaQBqJ/HfkYAeNdsmikNv6f3wxUqn+im/6+gt/2jYZltAAbo/4fo82uQTssvAzhdwLC7oLoTOgF4RB/PCCv7MgBAIoB/Q11X7M3m5ziHoPKMCP1/sD5+30B1826s12XKX3YD6KX/dwfwoP6/ug73hP5Nn4Tq8ljJfF9LNN2XdATkk8cPdLsA8SaA93VC+hWGC44+QdKgxh2YlnsBwGb9/+8AXjTMa6+XdYDqt5kGffOj5/cCsEn/PwB3V4B4K49lfwYwQv9vqVDQ0/B9NoBPChB2MfSNjP5eD1KAuOc+AFrri4a3/n4UwKv6fzuoG8HGFpZ7HcAqK+vcjLwLEI/lEa8k03ahKgKethLuCIB2+v+RANbnss4WAH6AunFMhbpgl89tG1AFh11m0/7E7YvuZgBTDfNyzRssrP84gI6G70+YzlGoAkSKMY8AcAFAizyOXY7jr6cNAXAij+X+D8Buw3fzQkGkYV4jANfzGxbqRu6U2XZ3wHoBwgHqRma7Pq5nocYfmOblVYBIB+BmmL8SwOv6/3P6uLibbfMrAJMt/E4Pme+rfAr+gbpOX4fqH38FwE9WwpXT83sYzys9bwOA0RaWeRiqEsLOMG2ZIW1EAFhsmEdQLf61DNNaQrVaWorTbwBeMnyvC5WPOuD2td3fMH8XDNdaK+uMgOUCRCp04dXKcg8AuGj4bl4oOGqYV0HHzTs/YaEKSub52newUoDQ8/vq43QDujBhmGdLASLYMH82gC/0/9sBvAVdMDCEmQhgkYXfqY/5vpbkR7owlR3fAOgNdQOz2GyeN1RJOs4wLQ6qphVQzdSnzeaZmGo44nVT2RWoGscqhRRv43ZNTe07ieiy3lZ7HX9rjM+evgk97iKfYc33P0ecxD2jP4CNfPthA0txuxuTN1TN3HELy1W3Mt1W5mn8Nd0N4apO4x64ncZz29bXUK0X0H+/sbZBZt7BzM8yc2WoG4xHoC46uW3DFznPfSBnPmG+L/nNG8zXH6enmSRyzj7PeZ3PuTE/5tWI6AdS3ayuQd3A5CdfKVeAsL5QNw9W42XEzBnMPJeZWwGoCOA/ACKIqE4u2zZKZOabhu/G49sNaszPKd3V5UE9vQaA8abfT/+GPsj5m4vC0ZWZK+pPVwDQXYeu688bzHwDwHNQvQLiiWgdEdXTy+d23p7mnOORcjtvK0PVdu81/OaRerolls5bU+WiSX6uw7lJYGbTwGoQUTkimq+7R12DquzMz3mLXOJiLawv1LmUYpif6z0BM3/DzP+COm9HAHifiP6V2zJmzO+/TOftQAANAEQT0S4i6qin1wDQy+y8bYGc+WmJkwJEGcHMcVADlTtC1TwZXYKqMahhmBYAVcMFAPFQmZNxnslpqNK4tyHzq8A2PErV1qib/tF9MH+EakmpyswVAWyEqjEpSvEA/A3fq1sLKMomnbaeBfAoqbEu5wG8CqAxETWGOkdSAdSysPhpK9MBVePkZvhezUIYYxp/GMB4HRdPncav4nYaz21bSwA8reNbH6r7UJ6YeTdUnhCaxzbOIWceAeTMJ3LsC/KfN5ivP0BPKwps9v0DqLg2ZOYKUBUtxZ2vADbmLcycwswfQ9Va19cFqzTkntYqmfVjzz6+zLyTmbtAFe5+hqpRBdRv+Lbh96vIzG7M/IMpKrbEVxQMM7/IzOX15z09bQMzt4MqyB2FaiUCcj9vq5Me06jldt5egmrtCzH85h7MbO1G29J5mwHV9a2wmae3cQBqAmiuz9vHimCb5uKhziXjS2htPW9vMfN3AA7hdn5ryzXC/P7LdN5GM3NPqPN2FoAVOl6noVogjOdtOWY2jZEoFeetFCDKlsFQ3SWMj0sFqycu/ABgGhG5E1ENAGOgbkig540iIn8i8gQwwbBsPNRN/CwiqqAHVNUiokeLIP7OUC0lFwFkElEnqH7LRe0HAIOJqC4RuQGYVAzbFMWrK9QjFBtADTJuAnUT/geAfrr2biGA2UTkS2qgcUtSg3y/BfA4ET1LajBrJSJqote7D0B3InIj9fSjwXnEwx3q4nsRgAMRvQXVfG4yH8A7pAb9EhE1IqJKAMDMZ6D6xH4D9XjPFFhAasD3UCKqor/Xg6p93pHHNtYDqENEvfV+PqeP18+WtlOAvGEZ1IDmykTkDdU0v8RK2MLmDnUhv0pqwHiej0YtBNsA2BPRcH08e0D1kbaIiF4lokdIDeZ3IKJBUK1i+3SQKAB9dNp8CqpLnpEdgClE5ERq0GYHAD/q9fUmogrMfAuqP7fpKTxfAhhBROE6LZQnos5EZGpFSYDq0iGKARFVJTWovRxUgfE6bv9W8wH8m4ia6d8qWF/Ld0Kl7XFE5Kh/+864XUjMQed1XwH40JBH+BHRE1aitQzAq0RUk4jKQ41R+J6L5wlJ7lCtA0k6j3qrqDfIzMehxohN1udSa5g9WMaIiAYRUUd9b2Wnz826UF25AHX+9tTndHOop+OZm6TP04ZQreLf63X3JSJv/ZtdhSoYZEFdA7oRUTudH7gQUVsiMrVAlIrzVgoQZQgzH2fmPVZmvwyVyZyAurAthbphAlRmsgHqAvUX7mzB6Ad1Y38Yqr/2j1C1I4WKma9A1QqvAnAZqp+yxZuXQt7uWgCfA9gK9aSc/+lZ9+Q7Au5T/aFqbE6xesrNeWY+D+ATqJsyB6ibygNQN+mXoWqt7Vg9vagj1IDny1AXhMZ6vR9C9T1PgOpilNdLijZAvR/lH6im6lTkbL6eDVWg3Qg1mHIBcj6S+WsADZFL9yWoPtRdABwgoutQ3RNWQXWJsboNZk6EeqDAa1D9eMcB6GTo8mVJfvKGdwHsAbAf6jj/pacVh8lQA9GvQg0eXVHUG2T1jpFuUN1RkqBandbDer6SCjW4NQGqlvgFAN116zIAjNLruwI1oHaN2fJnoPL4eKh0MoSZj+l5/QHEkeoGMhiqzzaYeSfUoO3PdRz/we1ucoC6WXybVDcJS0/8EoXLDur8OweV1zwK9dAHMPNyANOgrt3JUC2QXrrLTxeoAuMlAJ9BVYoczWU746EGRu/QaeK/UDe9liyEym+2QvVySIW6nygOs6G6eCZCjQf4JffghaYXVLfPRKi843tYP2+vQY1BPQ11Dr0HYBgz/6nnT4QaV3kFqnJyqYV1bIO6N9sINZj9dz29I4AjRJQMYCaA55g5ndX7fLrp9V2EerjFa7h9z/4Rbndxmp3vvS8k8iI5cd/RtQB/AXBmeS+FKEVIPapvCdRgWkmbZQwR7QXwETPnVgAUQpQiRLQCwD5mfqek41KWSAuEuC8QUTfdXFkJwHQAq+UGTZQmROQIYDTUG0YlbZYBpN5vUVV3XxgMVRO5saTjJYSwjoia6y5bdqQGLneCegy+yAcpQIj7xQio5t9jUE20I0o2OsKEiBaSeoHRQSvzidRLw2KIaD8ZXuR1ryCi+lBN4D5QzdOibKgP1WXrClQXpB7MXBSDT+9LkjeIIuIL1WUrGaqb6lBm3l+yUSp7pAuTEKJE6W4716GeZR5qYX5HqD65HQE8COBjZn7QPJwQ4t4ieYMQpZe0QAghShQzb4UaUGjN01A3EMzMOwBUJKJCH+QvhChdJG8QovSSAoQQorTzQ84nGZ2BvAhLCCF5gxAlxqGkI3A3vL29OTAwsKSjIUSptXfv3kv6bcVlmaUXgt3R95KIhgEYBgDlypVrVq9evTsWEkIokjcIISyxNW8o0wWIwMBA7Nlj7bUIQggiiss7VKl3Bjnf5OkPC284ZuYvoV6chQceeIAlbxDCOskbhBCW2Jo3lOkChBDivrAGwEgi+g5qoORV/ZbkMoO+tlRRWjDcv2w++EKOgSgCZT5vEKKskgKEEKJEEdEyAG0AeBPRGag3gzoCADPPg3q7b0eoN6veBDCwZGIqhChOkjfcH6RyoWySAoQQokQxc6885jPkvR1C3HckbxCi9JIChBBCCCFECZDad1FWyWNchRBCCCGEEDaTFgghhBCiGEhtsxDiXiEFCCGEEEIIIUpIWaxckAKEuO/R118X2rq4f/9CW5cQQgghRGlUZGMgiKg6EW0ioiNEdIiIRuvpXkT0KxEd03899XQiojlEFENE+4moaVHFTQghhBBCCFEwRTmIOgPAa8xcH0ALACOIqAGACQB+Y+baAH7T3wGgA4Da+jMMwOdFGDchhBBCCCFEARRZAYKZ45n5L/1/MoAjAPwAPA3A1GfkawBd9f9PA1jMyg4AFYnIp6jiJ4QQQgghhMi/YnmMKxEFAggDsBNAVdOr5vXfKjqYH4DThsXO6GlCCCGEEEKIUqLICxBEVB7ACgCvMPO13IJamHbHUHIiGkZEe4hoz8WLFxCbK5QAACAASURBVAsrmkIIIYQQQggbFGkBgogcoQoP3zLzSj05wdQ1Sf+9oKefAVDdsLg/gHPm62TmL5n5AWZ+oHLlykUXeSGEEEIIIcQdivIpTARgAYAjzDzbMGsNANOzLvsDWG2Y3k8/jakFgKumrk5CCCGEEEKI0qEo3wPxEIC+AA4Q0T497Q0A0wH8QESDAZwC8Iyetx5ARwAxAG4CGFiEcRNCCCGEEEIUQJEVIJh5GyyPawCAf1kIzwBGFFV8hBBCCCGEEHevWJ7CJIQQQgghhLg3SAFCCCGEEEIIYTMpQAghhBBCCCFsJgUIIYQQQgghhM2kACGEEEIIIYSwmRQghBBCCCGEEDaTAoQQQgghhBDCZlKAEEIIIYQQQthMChBCCCGEEEIImxXZm6iFEEKIsoy+/rqkoyCEEKWStEAIIYQQQgghbCYFCCGEEEIIIYTNpAAhhBBCCCGEsFmRjYEgooUAOgG4wMyhetr3AOrqIBUBXGHmJkQUCOAIgGg9bwczv1hUcRNCCJE76f8vhBDCmqIcRB0B4BMAi00TmPk50/9ENAvAVUP448zcpAjjI4QQQgghhLhLRdaFiZm3ArhsaR4REYBnASwrqu0LIcoGInqSiKKJKIaIJliYH0BEm4jobyLaT0QdSyKeQojiJXmDEKVXSY2BeBhAAjMfM0yrqTOBLUT0cAnFSwhRjIjIHsCnADoAaACgFxE1MAv2JoAfmDkMQE8AnxVvLIUQxU3yBiFKt5IqQPRCztaHeAABOhMYA2ApEVWwtCARDSOiPUS05+LFi8UQVSFEEWoOIIaZTzBzOoDvADxtFoYBmPIDDwDnijF+QoiSIXmDEKVYsRcgiMgBQHcA35umMXMaMyfq//cCOA6gjqXlmflLZn6AmR+oXLlycURZCFF0/ACcNnw/o6cZTQHwPBGdAbAewMvFEzUhRAmSvEGIUqwkWiAeB3CUmc+YJhBRZd1cCSIKAlAbwIkSiJsQoniRhWls9r0XgAhm9gfQEcA3RHRH3iWtk0LcUyRvEKIUK7ICBBEtA/AngLpEdIaIButZPXHn4OlHAOwnoigAPwJ4kZktDsAWQtxTzgCobvjujzu7IQwG8AMAMPOfAFwAeJuvSFonhbinSN4gRClWZI9xZeZeVqYPsDBtBYAVRRUXIUSptRtAbSKqCeAsVAVDb7MwpwD8C0AEEdWHukmQakQh7m2SNwhRismbqIUQJYaZMwCMBLAB6mWSPzDzISKaSkRddLDXAAzVLZTLAAxgZvOuDEKIe4jkDUKUbkX5IjkhhMgTM6+HGgBpnPaW4f/DAB4q7ngJIUqW5A1ClF7SAiGEEEIIIYSwmRQghBBCCCGEEDaTAoQQQgghhBDCZjIGQgghLKCvvy7pKAghhBClkrRACCGEEEIIIWwmBQghhBBCCCGEzaQAIYQQQgghhLCZFCCEEEIIIYQQNpMChBBCCCGEEMJmeT6FiYhaAngewMMAfACkADgIYB2AJcx8tUhjKIQQQgghhCg1ci1AENEvAM4BWA1gGoALAFwA1AHQFsBqIprNzGuKOqJCCCGEEEKUNHnMd95dmPoy82BmXsPM55g5g5mvM/NfzDyLmdsA2G5pQSJaSEQXiOigYdoUIjpLRPv0p6Nh3utEFENE0UT0RKHsnRBCCCGEEKJQ5VqAYOZLAEBE5YjITv9fh4i6EJGjMYwFEQCetDD9Q2Zuoj/r9TobAOgJIEQv8xkR2Rdkh4QQQgghhBBFx9ZB1FsBuBCRH4DfAAyEKiBYxcxbAVy2cf1PA/iOmdOY+SSAGADNbVxWCCGEEEIIUUxsLUAQM98E0B3AXGbuBqBBAbc5koj26y5OnnqaH4DThjBn9DQhRBlCRK2JaKD+vzIR1SzpOAkhhBCicNlcgNBPY+oD9fQlwIYnOFnwOYBaAJoAiAcwy7R+C2HZSkSGEdEeItpz8eLFAkRBCFEUiGgygPEAXteTHAEsKbkYCSGEEKIo2FqAeAXqpmAVMx8ioiAAm/K7MWZOYOZMZs4C8BVud1M6A6C6Iag/1NOfLK3jS2Z+gJkfqFy5cn6jIIQoOt0AdAFwAwCY+RwA9xKNkRBCCCEKnU2tCMy8BcAWw/cTAEbld2NE5MPM8fprN6j3SQDAGgBLiWg2AF8AtQHsyu/6hRAlKp2ZmYgYUA9fKOkICSGEEKLw5fUeiLWw0pUIAJj/n707j7drvvc//npLghiDhKaRiJp1QIWqDrTFxa+X6qDSQaIq9FJ01lbRVnvVNVRbVSmaUDUVlfamhqpQbg0xpeYhhqSCUCRBET6/P77fIys7+5yzzjn7nLX3Oe/n47EfZ6/5s9bZ67P3d32/67tijw6WPQ/YERguaS5wNLCjpC3zOh8FDszruVvShcA9wGLg4Ih4vUt7YmZVu1DS6cAwSQcAXyDVNJqZmVk/0lkNxAn578eBt7CkPfN4UgGgXRExvs7oMzuY/0ekh9WZWQuKiBMk7QwsADYBjoqIqyoOy8zMGsgPUTPopACRmy4h6YcR8cHCpD9Kuq5XIzOzlpGf23JFROwEuNBgZmbWj5W9iXpEvnEagNw1o+9gNjMAcpPDlyStXnUsZmZm1rvKdsX6FWCGpNl5eCz5/gUzs+zfwD8kXUXuiQkgIrrc4YKZmZk1r7K9MF0uaSNg0zzqvoh4pffCMrMW9L8seU6MmZmZ9VNdeRjc1qSah8HAFpKIiLN7JSozazkRMVXS8sDGedT9EfFalTGZmZlZ45UqQEg6h/QE6TuAtu5VA3ABwswAkLQjMJXUQ5uA0ZImRIQ7XDAzM+tHytZAjAM2j4h2nwlhZgPeicAuEXE/gKSNgfNItZdmZmbWT5Ttheku0nMgzMzaM6St8AAQEQ8AQyqMx8zMzHpB2RqI4cA9km4G3rx5uqMnUZvZgDNT0pnAOXn4s8CtFcZjZmZmvaBsAeKY3gzCzPqFLwEHA4eS7oG4DvhlZwtJ2hU4BRgEnBERx9WZZ29SHgrgzoj4TOPCNrNm5Nxg1rzKduN6raR1gG3yqJsj4uneC8vMWtBg4JSIOAnefDr1Ch0tkOc5FdgZmAvcImlaRNxTmGcj4NvA+yLiOUlr99YOmFlzcG4wa26l7oHIJfybgU8BewM3SfpkbwZmZi3namBoYXgo8JdOltkWeCgiZkfEq8D5wJ418xwAnBoRzwH44oXZgODcYNbEyt5E/V1gm4iYEBH7kk7s73W0gKSzJD0t6a7CuP+RdJ+kWZIulTQsjx8r6WVJd+TXr7q7Q2ZWmRUjYlHbQH6/UifLjALmFIbn5nFFGwMbS7pB0o25WYOZ9W/ODWZNrGwBYrmakv2zJZadAtSezFcB74iIdwEPkKoe2zwcEVvm10El4zKz5vGipHe3DUjaGni5k2VUZ1xtd9GDgY2AHYHxwBltFx+WWpE0SdJMSTPnz5/fpcDNrOk4N5g1sbI3UV8u6QpSn+4Anwb+3NECEXGdpLE1464sDN4IuBmUWf9xOHCRpCfy8EhSrujIXGB0YXhd4Ik689yYn2r9iKT7ST8abinOFBGTgckA48aN8zNrBqipqve7s5umTGncuqyrnBvMmlipGoiI+AZwOvAuYAtgckR8s4fb/gJLF0LWl3S7pGslfaCH6zazPhYRtwCbknpj+i9gs4jorBvXW4CNJK0vaXlgH2BazTx/AD4EIGk4qdnC7EbGbmZNx7nBrImVqoGQtD4wPSIuycNDJY2NiEe7s1FJ3wUWA+fmUfOAMRHxbG728AdJb4+IBXWWnQRMAhgzZkx3Nm9mDSRpG2BORDwZEa/lZkyfAB6TdExE/Ku9ZSNisaRDgCtIXTWeFRF3S/oBMDMipuVpu0i6B3gd+EZEPNvrO2ZmDZN7cvwx8NaI2E3S5sB7I+LMevM7N5g1t7L3QFwEvFEYfj2P6zJJE4CPAp+NiACIiFfaTvp8xfJh0pWEZUTE5IgYFxHjRowY0Z0QzKyxTgdeBZD0QeA44GzgBXKzgY5ExPSI2DgiNoiIH+VxR+UfCETy1YjYPCLeGRHn99qemFlvmUL6wf/WPPwAqdlju5wbzJpX2QLE4NyNGgD5/fJd3VjuIeFbwB4R8VJh/Ijc5zOS3kZqw+hqSLPWMKhQy/BpUhPHiyPie8CGFcZlZs1jeERcSL4YGRGLSRcjzawFlS1AzJe0R9uApD2BZzpaQNJ5wN+BTSTNlbQ/8AtgVeCqmu5aPwjMknQn8HvgoI6aPZhZUxkkqa055EeAvxamle2owcz6txclrUXuSUnSdqRaSjNrQWW/3A8CzpV0Kunknwvs29ECETG+zuj22jpeDFxcMhYzay7nAddKeobUbevfACRtiH8gmFnyVdJN0BtIugEYgXtiNGtZpQoQEfEwsJ2kVQBFxMLeDcvMWkVE/EjS1aRuW69su7eJVMP55eoiM7NmIGk5YEVgB2AT0jMe7s/dr5pZCyrbC1OXek8ws4ElIm6sM+6BKmIxs+YSEW9IOjEi3gvcXXU8ZtZzZe+BmEIXe08wMzMzy66U9AmpkU/6M7OqlC1AuPcEMzMz666vkrp/f1XSAkkLJS3zrCczaw1lCxDuPcHMOiTpEElrVB2HmTWfiFg1IpaLiCERsVoeXq3quMyse8r2wuTeE8ysM28BbpF0G3AWcEXhhmozG+Byd/AfzIMzIuJPVcZjZt1XqgYiIm4j9Z6wPXAg8PaImNWbgZlZa4mII0kPgTwTmAg8KOnHkjaoNDAzq5yk44DDgHvy67A8zsxaUKkChKRPAUMj4m7gY8AFkt7dq5GZWcvJNQ5P5tdiYA3g95KOrzQwM6va7sDOEXFWRJwF7JrHmVkLKnsPxPciYqGk9wP/AUwFTuu9sMys1Ug6VNKtwPHADcA7I+JLwNbAJyoNzsyawbDC+9Uri8LMeqzsPRBtPS79P+C0iLhM0jG9E5KZtajhwMcj4rHiyNwH/EcrisnMmsN/A7dLuob0ILkPAt+uNiQz666yBYh/Sjod2An4iaQVKF97YWYDw3TgX20DklYFNo+ImyLi3urCMrOqRcR5kmYA25AKEN+KiCerjcrMuqtsIWBv0oPkdo2I54E1gW/0WlRm1opOAxYVhl/ETR3NDJC0F/BSREyLiMuAf0v6WNVxmVn3lO2F6aWIuCQiHszD8yLiyt4NzcxajIrdtkbEG5Sv5TSz/u3oiHjz+VH5YuTRFcZjZj3Qq82QJJ0l6WlJdxXGrSnpKkkP5r9r5PGS9DNJD0ma5V6ezFrO7Hwj9ZD8OgyYXXVQZtYU6v3e8AUGsxbV2/cxTCF11VZ0BHB1RGwEXJ2HAXYj9SG/ETAJN30wazUHkZ4V809gLvAe0rlsZjZT0kmSNpD0NkknA7dWHZSZdU+vFiAi4joKN1Vme5K6gSX//Vhh/NmR3AgMkzSyN+Mzs8aJiKcjYp+IWDsi1omIz0TE01XHZWZN4cvAq8AFwEXAv4GDK43IzLqtVPWhpI8DPwHWJvWeINIzo1brxjbXiYh5pBXMk7R2Hj8KmFOYb24eN68b2zCzPiZpRWB/4O3Aim3jI+ILlQVlZk0hIl4ktziQNAhYOY8zsxZUtgbieGCPiFg9IlaLiFW7WXjoiOqMi2VmkiZJmilp5vz58xscgpn1wDnAW0gPm7wWWBdYWGlEZtYUJP1O0mqSVgbuBu6X5N4czVpU2QLEUw3sx/2ptqZJ+W9bE4e5wOjCfOsCT9QuHBGTI2JcRIwbMWJEg0IyswbYMCK+B7wYEVNJD558Z8UxmVlz2DwiFpCaLU8HxgCfrzYkM+uusgWImZIukDRe0sfbXt3c5jRgQn4/AbisMH7f3BvTdsALbU2dzKwlvJb/Pi/pHcDqwNjqwjGzJjJE0hBSAeKyiHiNOq0MzKw1lO1CbTXgJWCXwrgALuloIUnnATsCwyXNJfX5fBxwoaT9gceBT+XZpwO7Aw/lbe1XMjYzaw6Tc7fMR5IuCKwCfK/akMysSZwOPArcCVwnaT1gQaURmVm3lSpARES3fsxHxPh2Jn2kzryBe2Qwa0mSlgMWRMRzwHXA2yoOycyaSET8DPhZ27Ckx4EPVReRmfVEhwUISd+MiOMl/Zw6VY0RcWivRWZmLSMi3pB0CHBh1bGYWXOT9KeI+CiwuOpYzKx7OquBaLtxemZvB2JmLe8qSV8n9fP+ZveMEVH7LBgzG9hGVR2AmfVMhwWIiPhj/ju1o/nMzIC25z0UmyIGbs5kZku7veoAzKxnOmvCNBn4eUT8o860lYFPA69ExLm9FJ+ZtYiIWL/qGMysuUgaExGPF8f54ZJmra+zblx/CXxP0r2SLpL0S0lnSfob8H/AqsDvez1KM2t6kvat9yqx3K6S7pf0kKQjOpjvk5JC0rjGRm5mvegPbW8kXdyVBZ0bzJpXZ02Y7gD2lrQKMA4YCbwM3BsR9/dBfGbWOrYpvF+R1NvabcDZ7S0gaRBwKrAz6WGSt0iaFhH31My3KnAocFOjgzazXqXC+9LNGZ0brNGmSp3PVNaUKY1bV4sq243rImBG74ZiZq0sIr5cHJa0OnBOJ4ttCzwUEbPzMucDewL31Mz3Q+B44OuNidbM+ki0874zzg1mTazsk6jNzLrqJWCjTuYZBcwpDM+lpocWSVsBoyPiT40Nz8z6wBaSFkhaCLwrv18gaaGkjh4k59xg1sTKPonazKxDkv7IkiuMywGb0/lzIerVKb95lTI/oO5kYGKJ7U8CJgGsRQOqq11FbdZjETGom4v2Sm4YM2ZMN8Mxs6IuFSAkrRwRL3Y+p5kNQCcU3i8GHouIuZ0sMxcYXRheF3iiMLwq8A5ghlKB4C3ANEl7RMRSz6eJiMnAZID1pa40lTCz5tMruWHcuHHODWYNUKoJk6TtJd1DfrCcpC0k/bJXIzOzVvM4cFNEXBsRNwDPShrbyTK3ABtJWl/S8sA+wLS2iRHxQkQMj4ixETEWuBFY5geCmfU7zg1mTaxsDcTJwH+QT96IuFPSB3stKjNrRRcB2xeGX8/jtqk/O0TEYkmHAFcAg4CzIuJuST8AZkbEtPaWtWW5lxHrL5wbzJpb6SZMETFHS385vd74cMyshQ2OiFfbBiLi1XzlsEMRMR2YXjPuqHbm3bGnQZpZa3BuMGteZXthmiNpeyAkLS/p6+TmTF0laRNJdxReCyQdLukYSf8sjN+9O+s3s8rMl7RH24CkPYFnKozHzMzMekHZGoiDgFNIXajNBa4EDu7OBvMD6LaENx8U80/gUmA/4OSIOKGDxc2seR0EnCvpF3l4LtDpk6jNzMystZR9kNwzwGd7YfsfAR6OiMfUyLa7ZtbnIuJhYLv85HpFxMKqYzIzM7PGK9sL0/qSTpJ0iaRpba8GbH8f4LzC8CGSZkk6S9IaDVi/mfURST+WNCwiFkXEQklrSDq26rjMzMysscreA/EH4FHg58CJhVe35Zsr9yD10gJwGrABqXnTvPbWL2mSpJmSZs6fP78nIZhZY+0WEc+3DUTEc4DvZTIzM+tnyt4D8e+I+FmDt70bcFtEPAXQ9hdA0q+Buo+m9wNhzJrWIEkrRMQrAJKGAitUHJOZmZk1WNkCxCmSjibdPP1K28iIuK0H2x5PofmSpJERMS8P7gXc1YN1m1nf+y1wtaTfAAF8ATi72pDMzMys0coWIN4JfB74MPBGHhd5uMskrQTsDBxYGH28pC3zeh+tmWZmTS4ijpc0C9gJEPDDiLii4rDMzN707K239viBixPCjR/MyhYg9gLeVnxIVE9ExEvAWjXjPt+IdZtZdSLicuByAEnvk3RqRHSry2czMzNrTmULEHcCw4CnezEWM2txuRZxPPBp4BHgkmojMjOznta6LGXKlMaty1pW2QLEOsB9km5h6Xsg9mh/ETMbCCRtTOqSeTzwLHAB6TkQH6o0MDMzM+sVZQsQR/dqFGbWyu4D/gb8Z0Q8BCDpK9WGZGZmZr2l7JOor+3tQMysZX2CVANxjaTLgfNJN1GbmZlZP9Thg+QkXZ//LpS0oPBaKGlB34RoZs0sIi6NiE8DmwIzgK8A60g6TdIulQZnZmZmDdfZk6hXBoiIVSNitcJr1YhYrQ/iM7MWEREvRsS5EfFRYF3gDuCIisMyMzOzBuusCZM7OzazLouIfwGn55eZWVNYDNwOnFIYNxHYMf9tswWpKvVkUjeUbaYAkydP5sADlzyqatq0aWy99daMGjXqzXEHHHAAkydPZuutt+a229Izd0eOHMkTTzzBMcccw/e///0lKz2m5i/AnqQO9A8Hns/j1gO+D/wGKDYsPxl4FDRxScvR008/nUmTJqFC70sd7dOM/LfNYcDYPH+bHYD9AI4+Gh57LI0cNgx++lO49FK47LLCPh2z9F+APfeEvfaCww+H5/NOrbdeh/vU1X9U2zGIiGX+Tx3t09FA3iOGAT8FLgUKe7T0v2nixE726fvwm9/AtYWdOvlkePRROKWwUxMnwo47lv/wzaDTf9SkGyaV/uzNnDkTgHHjxtFVig4eiCJpLnBSe9Mjot1pfWHcuHHRtvNm3aWpUxu2rpgwoWHragRJt0ZE1zNDi1tfimN6uI6JDe2qcGLD1hQT2s/ZjeyqsbH7Dz4G0FfHoAznhu5r5IPkNLVxn1efFz4G0He5obMaiEHAKviGSDMzMzMzo/MCxLyI+EGfRGJmZmZmZk2vs5uoXfNgZmZmZmZv6qwA8ZE+icLMzMzMzFpChwWI3JOKmZmZmZkZUPJJ1L1B0qPAQuB1YHFEjJO0JnABqVOqR4G9I+K5qmI0MzMzM7OlVVaAyD4UEc8Uho8Aro6I4yQdkYe/VU1oZtaqGtHXOzNmQLGrvsMOg7Fj4SuFDrd32AH2269kv+g1f8F9vbuv96bv693MrJ4OnwPRqxtONRDjigUISfcDO0bEPEkjgRkRsUl76/BzIKwR/ByI/sfPgei5Vu3n3MegHOeG7vNzIKY0bF15jQ1bk49B3+WGzm6i7k0BXCnpVkmT8rh1ImIeQP67dmXRmZmZmZnZMqpswvS+iHhC0trAVZLuK7NQLmxMAhgzZkxvxmdmZmZmZjUqq4GIiCfy36dJzVC3BZ7KTZfIf5+us9zkiBgXEeNGjBjRlyGbmZmZmQ14lRQgJK0sadW298AuwF3ANKCtEfkElr63zcz6IUm7Srpf0kO584Ta6V+VdI+kWZKulrReFXGaWd9ybjBrXlXVQKwDXC/pTuBm4H8j4nLgOGBnSQ8CO+dhM+unJA0CTgV2AzYHxkvavGa220kdLrwL+D1wfN9GaWZ9zbnBrLlVcg9ERMwmdWJXO/5Z/PRrs4FkW+ChnBOQdD6pc9N72maIiGsK898IfK5PIzSzKjg3mDWxKnthMjMbBcwpDM/N49qzP/DnXo3IzJqBc4NZE6v6QXJmNrDV65i7bifWkj4HjCM9F6ze9Dd7aFurUdGZWVWcG8yamGsgzKxKc4HRheF1gSdqZ5K0E/BdYI+IeKXeioo9tK3aK6GaWR9ybjBrYi5AmFmVbgE2krS+pOWBfUi9sb1J0lbA6aQfCMt07Wxm/ZJzg1kTcwHCzCoTEYuBQ4ArgHuBCyPibkk/kLRHnu1/gFWAiyTdIWlaO6szs37CucGsufkeCDOrVERMB6bXjDuq8H6nPg/KzCrn3GDWvFwDYWZmZmZmpbkAYWZmZmZmpbkAYWZmZmZmpbkAYWZmZmZmpbkAYWZmZmZmpbkAYWZmZmZmpbkAYWZmZmZmpfV5AULSaEnXSLpX0t2SDsvjj5H0z/wwmDsk7d7XsZmZmZmZWceqeJDcYuBrEXGbpFWBWyVdlaedHBEnVBCTmZmZmZmV0OcFiIiYB8zL7xdKuhcY1ddxmJmZmZlZ11V6D4SkscBWwE151CGSZkk6S9IalQVmZmZmZmZ1VVaAkLQKcDFweEQsAE4DNgC2JNVQnNjOcpMkzZQ0c/78+X0Wr5mZmZmZVVSAkDSEVHg4NyIuAYiIpyLi9Yh4A/g1sG29ZSNickSMi4hxI0aM6LugzczMzMyskl6YBJwJ3BsRJxXGjyzMthdwV1/HZmZmZmZmHauiF6b3AZ8H/iHpjjzuO8B4SVsCATwKHFhBbGZmZmZm1oEqemG6HlCdSdP7OhYzMzMzM+saP4nazMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKcwHCzMzMzMxKa7oChKRdJd0v6SFJR1Qdj5n1rs7OeUkrSLogT79J0ti+j9LM+ppzg1nzaqoChKRBwKnAbsDmwHhJm1cblZn1lpLn/P7AcxGxIXAy8JO+jdLM+ppzg1lza6oCBLAt8FBEzI6IV4HzgT0rjsnMek+Zc35PYGp+/3vgI5LUhzGaWd9zbjBrYoOrDqDGKGBOYXgu8J6KYhkQNHVq5zN1QUyY0ND1tRpNbex3V0yIhq6vCZU559+cJyIWS3oBWAt4pk8iNLMqODeYNbFmK0DU+/W11C8oSZOASXlwkaT7ez2q8oYzwBOXJk4c6MegofuviT0ukKzXiDh6UafnfMl5lskNE6FnuWHixB4tXqNhn4sGfCbKaez+g48BNNcxcG7opomNreRops9EOf37vCinfx+DUrmh2QoQc4HRheF1gSeKM0TEZGByXwZVlqSZETGu6jiqNNCPwUDf/27o9JwvzDNX0mBgdeBftStybmhuPgY+Bl3k3DBA+Bi05jFotnsgbgE2krS+pOWBfYBpFcdkZr2nzDk/DWhrG/dJ4K8R0e/bdpkNcM4NZk2sqWogchvGQ4ArgEHAWRFxd8VhmVkvae+cl/QDYGZETAPOBM6R9BDp6uI+1UVsZn3BucGsuTVVAQIgIqYD06uOo5uasoq0jw30YzDQ97/L6p3zEXFU4f2/gU/1dVwN5s+FM9L8DgAAIABJREFUjwH4GHSJc8OA4WPQgsdAru0zMzMzM7Oymu0eCDMzMzMza2IuQHSRpLMkPS3prsK4n0iaJenswrjPSzqsmigbr539XlPSVZIezH/XyOM/IeluSX+TtFYet4Gk86uKvzu6uM+S9DNJD+XPwrvz+E0k3SrpTknvzeMGS/qLpJWq2TPrDc4NAyc3gPODlefc4NzQH3ODCxBdNwXYtW1A0urA9hHxLmCQpHdKGgpMBH5ZSYS9YwqF/c6OAK6OiI2Aq/MwwNeA7YCzgc/kcccC3+v9MBtqCuX3eTdgo/yaBJyWxx+Y5/kk8PU87kvAORHxUq9FblWYgnNDm/6eG8D5wcqbgnNDG+eGfpIbXIDoooi4jqX7mX4DWF6SgKHAa8A3gJ9FxGsVhNgr6uw3wJ5A26OspwIfy+/fAFYAVgJek/QBYF5EPNgXsTZKF/d5T+DsSG4EhkkaSfo8DGXJsRgG/CcpSVo/4tywlH6dG8D5wcpzbliKc0M/yQ1N1wtTq4mIhZIuBm4nlSpfALaJiB9UG1mfWCci5gFExDxJa+fx3yd1vfcE8DngQvpP93rt7fMoYE5hvrl53KmkE34F0hWFo4Afua/y/s+5YcDlBnB+sBKcG5wb+kNucAGiASLieOB4AElnAEdJ+iKwCzArIo6tMr6+FhFXAVcBSJpA6oZvE0lfB54DDmuWKrgGqvfs+IiIx4EdASRtCLwVuE/SOcDywPci4oE+i9L6lHPD0gZobgDnB6vh3LA054altERucBOmBpK0VX77ALBvROwNvEPSRhWG1ZueylVt5L9PFyfmG30mkNp0/jfwBeBW4LN9HGcjtbfPc4HRhfnWJV1JKfoRqT3nocC5wNH5Zf2cc8OAyA3g/GBd5Nzg3ECL5gYXIBrrh6RqpiGkJ2dCatfXFHfM94JppBOd/PeymunfBE7JbTqHAkHrH4/29nkasG/uUWE74IW26koASTsA/8ztOVciHYfXae1jYeU5NyytP+YGcH6wrnNuWJpzQ6vkhojwqwsv4DxgHukGl7nA/nn8x4CjC/OdAPwDOLfqmHtrv4G1SO03H8x/1yzM/1bgT4XhTwF3AzcAI6ren0bvM6ka8lTg4fx/H1dYj0hVs2vk4c2A24BZwPuq3k+/eu/zksc7N/Sz3NDV/XZ+GNgv5wbnhv6YG/wkajMzMzMzK81NmMzMzMzMrDQXIMzMzMzMrDQXIMzMzMzMrDQXIMzMzMzMrDQXIMzMzMzMrDQXIFqEpLUk3ZFfT0r6Z2F4+ZLr+I2kTTqZ52BJDXlgi6Q9c3x3SronP2Wzo/k/nPtBrjdtpKTphXVNy+NHS7qgEfGatSLnBucGs3qcG5wbepO7cW1Bko4BFkXECTXjRfqfvlFJYEvHsgLwCKlP4yfy8HrRwaPXJR0LPBMRP60z7Uzgtog4NQ+/KyJm9VL4Zi3JucG5wawe5wbnhkZzDUSLk7ShpLsk/Yr0gJGRkiZLminpbklHFea9XtKWkgZLel7Scblk/ndJa+d5jpV0eGH+4yTdLOl+Sdvn8StLujgve17e1pY1oa1OegjKvwAi4pW2JCBpHUmX5OVulrSdpA2ALwLfyFcftq9Z30jSA1nI65tV2P878vvfFK6uPCPpu3n8EXk7s4rHw6w/c25wbjCrx7nBuaERXIDoHzYHzoyIrSLin8ARETEO2ALYWdLmdZZZHbg2IrYA/g58oZ11KyK2Bb4BtJ1EXwaezMseB2xVu1BEPA1cATwm6XeSxktq+7z9DDg+x7g3cEZEPAycAfxPRGwZEf9Xs8pfAFMl/VXSdySNrLPN/SJiS2Av4BngbEm7A2OA9wBbAtvXSTJm/ZVzA84NZnU4N+Dc0BMuQPQPD0fELYXh8ZJuI11Z2IyUKGq9HBF/zu9vBca2s+5L6szzfuB8gIi4k/So+WVExERgZ2AmcAQwOU/aCfhVvgLwB2ANSUPb3z2IiOnABsCZeX9ul7RW7Xx5PRcBX4qIOcAuwG7A7aTjsSGwcUfbMutHnBsy5wazpTg3ZM4N3TO46gCsIV5seyNpI+AwYNuIeF7Sb4EV6yzzauH967T/WXilzjwqG1iuMpwl6XfAvaTqRuX4ijEgdbzaiHgWOBc4V9LlpIRUm4R+DZwfEdcUYj02Is4sG7NZP+LcsIRzg9kSzg1LODd0g2sg+p/VgIXAglxd9x+9sI3rSVWISHonda5USFpN0gcLo7YEHsvv/wIcXJi3rR3kQmDVehuU9JG2qw2SVgPWBx6vmecwYEjNTWJXAPtLWjnPs66k4SX306w/cW5wbjCrx7nBuaHLXAPR/9wG3APcBcwGbuiFbfyc1E5wVt7eXcALNfMI+LakXwMvA4tY0l7yYOA0SfuRPoPX5HGXARdJ+jhwcE17xm2AX0h6jVTwPS0ibpe0YWGerwMvtd0cBfwiIs6QtClwY75SsRD4DKmto9lA4tzg3GBWj3ODc0OXuRtX6zJJg4HBEfHvXPV5JbBRRCyuODQzq5Bzg5nV49zQ/7gGwrpjFeDqnBAEHOgkYGY4N5hZfc4N/YxrIMzMzMzMrDTfRG1mZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5AGFmZmZmZqW5ANHPSRorKSQNLjHvREnX90VcnW1b0iJJb+vGej4r6crGRmdmlkh6WNJ7q47DzLpG0l8lfbrqOPoLFyCaiKRHJb0qaXjN+DtyIWBsNZEtVRBZlF+PSjqit7YXEatExOySMQ0uLHduROzSW3FZ/yRphqTnJK1QdSy9RdKeOZcskPSMpKurzCmNJOnuQm56XdK/C8Pf6cF6z5d0ZHFcRGwQEX/vedTLbGtFST+T9M8c92xJPym57HGSzmh0TNa78vfoy4XP6iJJb606rr4k6c+FfX8t/wZqG/5VD9a7zDkRER+OiAt6HvUy25Kko/P/c5GkOZLOLrnsQZL+0uiY+kKnV6Wtzz0CjAd+DiDpncDQSiNa2rCIWJyvwF0t6Y6IuLw4g6TBEbG4ovjMuiT/iP4A8AKwB3BRH267T84VSRsCZwMfB/4KrALsArzRwG0IUEQ0bJ1lRcTbC3HMAH4bEa32g/poYDPg3cDTwPqAazr6v/+MiMp/QEoaFBGv9/V2I2K3QgxTgLkRcWT7SzSlScAngA9FxCO5ELh7xTH1OtdANJ9zgH0LwxNIX/xvkrS6pLMlzZf0mKQjJS2Xpw2SdEK+wjgb+H91lj1T0rx8petYSYO6GmS+Anc38I683pB0sKQHgQfzuE0lXSXpX5Lul7R3IY61JE3LV0NvBjaoiTPyjx4kDZV0Yt7XFyRdL2kocF2e/flc6n+vlm0KFbmE/2C+wnxq/qHTdqxOzMfqEUmH1NZo2ICwL3AjMIV0vr2pg88ekt4v6f8kPZ+vOE3M42dI+mJhHfU+k7Xnyil5HQsk3SrpA4X5B0n6jlLTmYV5+uj8WT6xJt4/Sjq8zj5uCTwSEVdHsjAiLo6IxzvaRp62vaRb8v7fImn7wvZmSPqRpBuAl4C3dSXHSFpB0k8lPZFfP1WuBZK0o6S5kr4m6em8vv06/le2T9KBOQ/9S9L/ShpV2Pdf5Hz6gqQ7JW0i6VDSj4Lv5fxyUZ7/SUnvz++Pk3SupPPycZslacvCNrfN61so6XeSLlFNjUbBNsDFEfFU/h/NjohzC+saLemynK9mSzooj/8Y8FVgQo7z5u4eI2teOY/Mzp+lRyR9tjDtAEn35mn3SHp3Hr9ZPkefV6ql26OwzBRJp0maLulF4EP5fDxB0uOSnpL0q7Z8Vyee5ZR+ezyWz8+zJa2ep7W1DpiQ1/WMpO/2YN/3yufW85L+JmnzwrTv5dywIB+DD7R3Tki6UdLn8vuDlGphf5bX+7CknQrr3VDSDfmYXi7pdLVfy7cNMD0iHgGIiCeKFzAkrZmPz5NKef7ofPy2An4K7JjjfLK7x6gSEeFXk7yAR4GdgPtJV6IGAXOA9YAAxub5zgYuA1YFxgIPAPvnaQcB9wGjgTWBa/Kyg/P0PwCnAysDawM3AwfmaROB69uJbWzbegAB7yP9YPhInh7AVXmbQ/P65wD75WXeDTwDvD3Pfz5wYZ7vHcA/i9vO69swvz8VmAGMysdke2CFYkyF5SbWWc+fgGHAGGA+sGvhWN0DrAusAfyldn1+9f8X8BDwX8DWwGvAOoVp7X32xgALSbWFQ4C1gC3zMjOALxbWUe8z+ea5ksd9Lq9jMPA14ElgxTztG8A/gE3yubdFnndb4AlguTzf8HxOrlNnH98G/Bs4GfgQsErN9Pa2sSbwHPD5HNv4PLxWYV8fB96epw+hgxxTJ64fkApvawMjgP8Dfpin7QgszvMMIV3RewlYo5P/51LHP4/bB7gX2Div61jgmjxtT+DvwGqki2pvB9bO084HjqxZ15PA+/P743JMO+fPx8nAjDxtxfz/OSgfm31In68j24n7WFIN9EHkPFmYNij/f74FLJ/343Fgh0IcZ1R9LvnVtRf5O7/EfCsDC4BN8vBIlnyXfor0/blNPnc3JP1mGELKbd/Jn5kPk3JW2zqmkGpd35c/9yuSfsxOy+f9qsAfgf9uJ6Yv5PW/jVSjeQlwTp42lpTnfk36PbAF8AqwWSf7OQU4tmbcdsA8Un4eRLra/0A+p7YAZgPr5H1/G7B+Xm6Zc4KUaz6X3x+Uz8d983q/AjxamPc24Ef52O0IvNjeOQZ8kfTb4quk3zqDaqb/mdSqZKX8v7sdmFCI4y9Vfxa79fmtOgC/Cv+MJQWII4H/BnYl/dAYnE/GsfmD/gqweWG5A1nypfVX4KDCtF1Y8sN/nbzs0ML08Sz5Ip1I5wWI50k/IO4FDi1MD+DDheFPA3+rWcfppGr6QfnE3bQw7cfUKUCQEtvLwBYdxNRZAeL9heELgSMKx+rAwrSdatfnV/9+Ae/Pn8Xhefg+4Cv5fUefvW8Dl7azzhl0XoD4cCdxPde2XdIFhT3bme9eYOf8/hDSVbD21rld/vzPJxUmppALEu1tg1RwuLlm3N+BiYV9/UFhWoc5ps76HwZ2Lwz/B/lLnPSl/XLN+f00sF0nx26p45/HXQN8tjA8JP/f1yEVTO4mFchUs1yZAsSfCtPeDTyf3+8CzK5Zdmbt+mpiOiwf31eAucD4PG0H4MGa+b8PnFaIwwWIFnuRvvMXkb5Xnwf+0M58K+fpnyieW3naFcBhdZb5QP6sLlcYdx5wTH4/BTi7ME2kH8kbFMa9l1RzWS+mq4H/Kgxvks+pwSz5bl63MP1mYJ9OjscUli1A/Ab4bs24x4D3kAr780gXRQbXzFOmAHFXYdqaOeZhpAL6y8AKhem/b+8cy8duQs4zL5EulrZ9j6yXj+uQwvz7AX8uxNGSBQg31WhO55Ca56xPTfMl0lXG5UknUJvHSFdIAd5KuvJfnNam7arEPKVWPJB+JBXn78zwaL/NdnE96wHvkfR8Ydxg0r6NyO/bi3Op7ZGujDzchRhrFasFXyJdLYFlj1VXjoP1DxOAKyPimTz8uzzuZDr+7I1uZ3xZS33WJH2NdBXrraQvsdXy9jvb1lRS7cVV+e8p7W0wIm4E9s7b2wa4APguqTDU3jbeyrLnZjHf1O5LV3NM7fofy+PaPFuTb4rnb1esB/xK0qmFcYtJtY9/BjYlXeAYJen3wDcjYlHJdXeUX+bWzNtujomI10j/v1MkrUT6YXF2bn6xHjC2Jp8OItWaWmv7WNTcA6F08/Dn8uCPI+LHSr0HfR04U6nJ4Ncioq21QXvn7pxY+p6kjs7dEaQr5LcWzl2RPmf11Dt32y5Utmnv3OiK9YC9JX2jMG55YFREXKLUmcuPgE0l/Rn4akQ8VXLdtfGRY3wrMD8iXilMn0OqlVlGpJLAVGCqpOWBT+b3t5Hy+YrA/Jqc+FDJGJuW74FoQhHxGKkqe3dStWDRM6RS/nqFcWNIVZiQSuOja6a1mUO6sjU8Iobl12pRuAGxp6HXbOvawnaGRepZ6UukK6CLO4iz6BnS1dIN6kyLOuO6Yh7pB0Sb0e3NaP1Pbtu7N7BDbpv6JKkaewtJW9DxZ29OO+MhXW1aqTD8ljrzvPnZVbrf4Vs5ljUiYhipaUHbt01H2/otsGeOdzNS86FORcQtpNzyjk628QRL5xpYOt8stS90PcfUrn9MHtdoc0i1JsV8NDQibo3kpIjYCngXqVnEYXm5nuSY2vwCJXNMRLwUESeRjuWmOf77auJfNSL2akCc1mQi4qD8fblKRPw4j7siInYmNYG5j9Q8CDo+d0cr3x+ZdXTuPkO66v72wmds9Yho70d/vXN3MVD2x3tZc4Cjaj77K0XEJQARMTUitic1X1qR1BQQen7ujtDSvfKVPXdfjYjfkWp135HjX0TO7YWc+O4GxFkpFyCa1/6kZg4vFkdG6iXhQuBHklaVtB6p3d1v8ywXAodKWlfSGsARhWXnAVcCJ0paLd/Es4GkHXoh/j8BG0v6vKQh+bWNpM3yPlwCHCNppXxD1IR6K8lXT84CTpL0VqUbHt+bT+z5pF5kuvy8iOxC4DBJoyQNI/2Is4HjY8DrwOakm4y3JP0I/xuwbyefvXOBnSTtLWmwUqcAbTfP3gF8PH+2NySdyx1ZlfTFOx8YLOkoUg1EmzOAH0raSMm7JK0FEBFzgVtINXsXR8TL9TagdMP3AZLWzsObknqcurGTbUwnncefyfv56Xy8/lRvO93IMecBR0oaodR99VEsyWWN9Ku8nU0AJK0h6RP5/XaSxil1nvAi8CrpcwHpx1B388t1wFBJk/Kx25tUOKlL6WbxDyh15zpE0iTS1d87gevzPIfn6YPz/6jtR8hTwPoqXOK0/kPSOpL2kLQyqVC5iCWf0TOAr0vaOp+7G+bfBTeRPs/fzJ+nHYH/JDXLW0bOd78GTi7kiVGS/qOdsM4DviJpfUmrkJohX9BBC4Xumgx8OZ+jkrRKPhYrSdpc0g45J7+cX8Vzt7vnxAOkQtqR+dh9kNSkvC5JX5S0a45tOaWb1TckNf98hJRnj8+/2ZbLefb9hThHSxrSjTgr5QJEk4qIhyNiZjuTv0xKDLNJXyy/I/3QgZQAriB96dzGsjUY+5Kq/+4htbP+PemKRkNFxEJSG+B9SFcqngR+QroBFVJ77VXy+Cmkdo7t+TrpBsJbgH/l9SwXES+Rqi5vUOpFYbsuhvlr0o+dWaSbmqaTfsj1eVd2VokJwG8i4vGIeLLtBfwC+Gz+QdneZ+9xUg3h1/L4O1jy4/Bk0o/Qp0jV2ufSsStIzWgeIDUD+DdLNy04iVTYvZJ0I+WZLN2181TgnaRCRHueJxUY/iFpEXA5cClwfEfbiIhngY/m/XwW+Cbw0UKTr3q6kmOOJd0XMIt0nG9jyRXEhomI80j/10skLSD9v3bOk4eRctDzpJz6GPCzPG0ysE3OL3V/eHWwzZdJ3eZ+mXQcPkb6X7/SziKv5O0+RbrXYz9S85a5uXnT7qSb+B8jFTZPY0mTkPNJtV7/kvR/XYnTWsJypHPwCVK+2YHU8QMRcRHpe/B3pJuk/wCsGRGvks753Ui1C78kXRi5r4PtfIvUtObGfJ78hXRvQz1nsaS59SOkvPXl7u9ifRFxA3AoqYnh86Q8+RnSlfuhwImk/ZtHOh+Oyot2+5zITZL2Id0X+RzpRvSLaP/cXUi6v3Nunv+HpI5tbsnTx5PyzH2k/98FLGnqdTnpXpinJdU2eWxqSsfJzCTtBvwqImqbbJg1rXx17LekXtr6/BkMVp6kO4HjcoHGzFqEpMuAGyPiv6uOpVm4BsIGLKU+/nfPzQFGka4gXFp1XGZl5Wrvw0i9g7jw0GQkfUjS2oUmSRuQbng3syYm6T1Kz7NYTtJ/kpowTas6rmbiAoQNZCJ1hfgcqQnTvSyp/rQ+IukspQcR3dXOdCk97OchpYcJvbvefAONpM1IVfojSf23W/N5O3AXKcf8F/DxTpp/WYFzg1VoXVIT8UXA/wBfiIi7qw2pubgJk5lVKjfBWUTqk/wddabvTmpbuzup7+9TIuI9fRulmfU15waz5uUaCDOrVERcR7qxrD17kn5ARH6WwTBJDb/x38yai3ODWfNyAcLMmt0olu6VaC5LPwzJzAYm5wazirT0k6iHDx8eY8eOrToMs6Z16623PhMRI6qOo4fq9eO9TNvLfJPqJICVV15560033bS34zJrWc4NZlZP2dzQ0gWIsWPHMnNme49KMDNJj1UdQwPMZemngK5LnacVR8RkUr/9jBs3LpwbzNrn3GBm9ZTNDW7CZGbNbhqwb+5xZTvghfzEYzMb2JwbzCrS0jUQZtb6JJ0H7AgMz0/iPBoYAhARvyI9IXx30hNSXyI9odfM+jnnBrPm5QKEmVUqIsZ3Mj2Ag/soHDNrEs4NZs2rkiZMklaUdLOkOyXdLen7efz6km6S9KCkCyQtX0V8ZmZmZmZWX1X3QLwCfDgitgC2BHbN7Rd/ApwcERuRnty5f0XxmZmZmZlZHZUUIPJDXxblwSH5FcCHgd/n8VOBj1UQnpmZmZmZtaOyXpgkDZJ0B/A0cBXwMPB8RCzOs/iBMGZmZmZmTaaym6gj4nVgS0nDgEuBzerNVjui+ECYMWPG9GqMNjBo6tSGrSsmTGjYuszMzMyaUeXPgYiI54EZwHbAMElthZp2HwgTEeMiYtyIEa3+EE0zMzMzs9ZSVS9MI3LNA5KGAjsB9wLXAJ/Ms00ALqsiPjMzMzMzq6+qJkwjgamSBpEKMRdGxJ8k3QOcL+lY4HbgzIriMzMzMzOzOiopQETELGCrOuNnA9v2fURmZmZmZlZG5fdAmJmZmZlZ63ABwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMSnMBwszMzMzMShtcdQBmZmbNSFOnNniNExu2ppgQDVuXmVlXuQBhZmbL8I9nMzNrj5swmZmZmZlZaS5AmFmlJO0q6X5JD0k6os70MZKukXS7pFmSdq8iTjPrW84NZs2rz5swSRoNnA28BXgDmBwRp0g6BjgAmJ9n/U5ETO/r+Mys70gaBJwK7AzMBW6RNC0i7inMdiRwYUScJmlzYDowts+D7QFNVcPW5eY7NhAMlNxg1qqquAdiMfC1iLhN0qrArZKuytNOjogTKojJzKqxLfBQRMwGkHQ+sCdQ/JEQwGr5/erAE30aoZlVwbnBrIn1eQEiIuYB8/L7hZLuBUb1dRxm1hRGAXMKw3OB99TMcwxwpaQvAysDO/VNaGZWIecGsyZW6T0QksYCWwE35VGH5HaMZ0lao7LAzKyv1GvbU9tGZzwwJSLWBXYHzpG0TO6SNEnSTEkz58+fXzvZzFqLc4NZE6usACFpFeBi4PCIWACcBmwAbEmqoTixneWcCMz6j7nA6MLwuizbDGF/4EKAiPg7sCIwvHZFETE5IsZFxLgRI0b0Urhm1kecG8yaWCUFCElDSIWHcyPiEoCIeCoiXo+IN4Bfk9o/LsOJwKxfuQXYSNL6kpYH9gGm1czzOPARAEmbkX4k+OqBWf/m3GDWxPq8ACFJwJnAvRFxUmH8yMJsewF39XVsZta3ImIxcAhwBXAvqUeVuyX9QNIeebavAQdIuhM4D5gYEe6KyKwfc24wa25V9ML0PuDzwD8k3ZHHfQcYL2lLUhvHR4EDK4jNzPpY7q55es24owrv7yHlDTMbQJwbzJpXFb0wXU/9m6P8zAczMzMzsybnJ1GbmZmZmVlpLkCYmZmZmVlpVdwDYWbW9DR1atUhmJmZNSXXQJiZmZmZWWkuQJiZmZmZWWkuQJiZmZmZWWkuQJiZmZmZWWkuQJiZmZmZWWnd7oVJ0nuBzwEfAEYCLwN3Af8L/DYiXmhIhGZmZmZm1jS6VQMh6c/AF4ErgF1JBYjNgSOBFYHLJO3RqCDNzMzMzKw5dLcG4vMR8UzNuEXAbfl1oqThPYrMzMzMzMyaTrdqINoKD5JWlrRcfr+xpD0kDSnOY2ZmZmZm/UdPb6K+DlhR0ijgamA/YEpPgzIzMzMzs+bU0wKEIuIl4OPAzyNiL9K9EGY2AEl6v6T98vsRktavOiYzMzNrrB4XIHJvTJ8l9b4EPejZycxal6SjgW8B386jhgC/rS4iMzMz6w09LUAcTvqxcGlE3C3pbcA1PQ/LzFrQXsAewIsAEfEEsGqlEZmZmVnD9ai2ICKuBa4tDM8GDu1pUGbWkl6NiJAUkDpZqDogMzMza7xuFSAk/RGI9qZHhJ8BYTbwXCjpdGCYpAOALwC/rjgmMzOzhtLUqQ1dX0yY0ND19YXu1kCckP9+HHgLS9o5jwce7WFMZtaCIuIESTsDC4BNgKMi4qqKwzIzswbyj+fG01Q1bF0xod3r+w3VrQJEbrqEpB9GxAcLk/4o6bqGRGZmLUPSIOCKiNgJcKHBzMysH+vpTdQj8o3TAOQuG0f0cJ1m1mIi4nXgJUmrVx2LmZmZ9a6edrn6FWCGpNl5eCxwYEcLSBoNnE1q+vQGMDkiTpG0JnBBXsejwN4R8VwP4zOzvvNv4B+SriL3xAQQEe5YwczM6mrF5jvW816YLpe0EbBpHnVfRLzSyWKLga9FxG2SVgVuzT84JgJXR8Rxko4AjiD1KW9mreF/WfI8GDMzM+unGvHQt61JtQaDgS0kERFntzdzRMwD5uX3CyXdC4wC9gR2zLNNBWbgAoRZy4iIqZKWBzbOo+6PiNeqjMnMzMwar0cFCEnnABsAdwCv59FBaqJUZvmxwFbATcA6uXBBRMyTtHZmcOnmAAAd5klEQVRPYjOzviVpR1Lh/1FAwGhJEyLCHSuYmZn1Iz2tgRgHbB4RXW50JmkV4GLg8IhYIJVrAydpEjAJYMyYMV3drJn1nhOBXSLifgBJGwPnkWopzczMrJ/oaS9Md5Fuhu4SSUNIhYdzI+KSPPopSSPz9JHA0/WWjYjJETEuIsaNGOEOn8yayJC2wgNARDwADKkwHjMzM+sFPa2BGA7cI+lm4M2bpzt6ErVSVcOZwL0RcVJh0jRgAnBc/ntZD2Mzs741U9KZwDl5+LPArRXGY2ZmZr2gpwWIY7qxzPuAz5O6e7wjj/sOqeBwoaT9gceBT/UwNjPrW18CDgYOJd0DcR3wy84WkrQrcAowCDgjIo6rM8/epHwTwJ0R8ZnGhW1mzWgg5AZ3YWqtqqfduF4raR1gmzzq5oio2/SosMz1pB8X9XykJ/GYWaUGA6e01Szmp1Ov0NECeZ5TgZ2BucAtkqZFxD2FeTYCvg28LyKecwcLZv2fc4NZc+vRPRC55H8zqbZgb+AmSZ9sRGBm1nKuBoYWhocCf+lkmW2BhyJidkS8CpxP6tK56ADg1LYHS3Z2kcLM+gXnBrMm1tMmTN8Ftmk7aSWNIP1g+H1PAzOzlrNiRCxqG4iIRZJW6mSZUcCcwvBc4D0182wMIOkGUlOGYyLi8gbEa2bNy7nBrIn1tACxXE2J/1l63rOTmbWmFyW9+/+3d+/RklTl3ce/P0GQIBdFxZG7QDBEI8rBC+ZVE+95jYhXMFFGMaMuQdRoQi7clCRINHgjhFFgBoOiRNGJaxQMETW+arjpBFAEEWUARVAEQVHgef+oOtBzcs5Mz+k+p7rPfD9r9eqq3VXVT9V0P3N27117V9XFAEn2Bn65jn2m6844tSPvxsDuNBNNbg98Jcmjq+qWNQ7kEM8Clvc5JHhfli0b3rG0vswN0ggbtALx+STn0Iz1DvBy4HMDHlPSeHozcFaS69v1RTQ5YW1WAzv0rG8PXD/NNl9vZ7X+fpIraP5ouKB3o6paCiwFmJiY8G5CabyZG6QRNuhN1G9P8iLg92l+LVhaVWcPJTJJY6WqLkjyKGAPmnzwnfY/9rW5ANg9yS7AdcABwNRRVD4NHAgsS/IQmm4LVw81eEmjxtwgjbBBb6LeBVhZVW+tqrfQtEjsPIzAJI2HJPskeThAW2F4PHAs8J4kD17bvlV1F3AIcA7wbeATVXVZknckmZxP5hzg5iSXA18E3l5VN8/R6UiaA0m2TXJKks+163u2w7ZPy9wgjbZBuzCdBezbs353W7bP9JtLWoBOBp4JkOSpNHO6HArsRdNtYK0js1XVSmDllLIje5YLeGv7kDSelgGn0Qy+AvBd4OM0E8tOy9wgja5Bb3jeuB1eDYB2eZMBjylpvGxUVT9tl19O05Xxk1V1BLBbh3FJGh0PqapPAPfAvS0Md3cbkqTZGrQC8ZOepkSS7AfcNOAxJY2XjZJMtmY+A/jPntcGbeWUtDDcnmQb2pGUkjwJ+Hm3IUmarUH/c389cEaSE2mSwmrgVQNHJWmcfAz4UpKbaIZt/QpAkt3wDwRJjbcCK4Bd23kbHso6ujdKGl2DjsL0PeBJSR4IpKpuG05YksZFVf1dkvNohm09t+2XDE0L56HdRSZpFCS5H/AA4GncN0rbFX2M0iZpRA1UgUiyLfD3wCOq6nlJ9gSeXFUz3hQlaeGpqq9PU/bdLmKRNFqq6p4k76mqJwOXdR2PpMENeg/EMpph1B7Rrn+XZjIpSZKkSecmeXEyzKnCJXVl0AqEoypIkqR1eSvNMO+/TnJrktuS3Np1UJJmZ9AKhKMqSAIgySFJHtR1HJJGT1VtUVX3q6r7V9WW7fqWXcclaXYGHYXJURUkTXo4cEGSi4FTgXN6bqiWtIFrh31/art6flV9tst4JM3eQC0QVXUxzagK+wKvA363qlYNIzBJ46Wq/hbYnWZm2cXAlUn+PsmunQYmqXNJjgMOAy5vH4e1ZZLG0EAViCQvBTarqsuAFwIfT/L4oUQmaey0LQ4/ah93AQ8C/i3J8Z0GJqlrfwQ8q6pOrapTgee2ZZLG0KD3QBxRVbcl+X3gOcBy4KTBw5I0bpK8KclFwPHAV4HHVNUbgL2BF3canKRRsHXP8ladRSFpYIPeAzE54tL/BU6qqs8kOXrAY0oaTw8BXlRVP+gtbMeAf35HMUkaDf8AXJLkizQTyT0V+KtuQ5I0W4NWIK5LcjLwTOBdSTZl8FYNSeNpJfDTyZUkWwB7VtU3qurb3YUlqWtV9bEk5wP70FQg/rKqftRtVJJma9A/9l9GM5Hcc6vqFuDBwNvXtVOSU5PcmOTSnrKjk1yX5Jvtw76R0ng5CfhFz/rt2KVREpBkf+COqlpRVZ8BfpXkhV3HJWl2Bh2F6Y6q+lRVXdmu31BV5/ax6zKaG6imOqGq9mofKweJTdK8S++wrVV1D4O3ckpaGI6qqnvniWp/dDyqw3gkDaCT7kZV9WV6ujpIWhCubm+kvn/7OAy4uuugJI2E6f7e8AcGaUyN2v0KhyRZ1XZxckZbaby8nmZOmOuA1cATgSWdRiRpVFyY5J+S7JrkkUlOAC7qOihJszNKFYiTgF2BvYAbgPdMt1GSJUkuTHLhT37yk/mMT9JaVNWNVXVAVT2sqratqldU1Y1dxyVpJBwK/Br4OHAW8CvgjZ1GJGnWBmo+TPIi4F3Aw2hGVQjNXFJbru+xqurHPcf9EDDtFPdVtRRYCjAxMVHTbSNp/iV5AHAw8LvAAybLq+o1nQUlaSRU1e3A4QBJNgI2b8skjaFBWyCOB15QVVtV1ZZVtcVsKg8ASRb1rO4PXDrTtpJG0keAh9NMKvklYHvgtk4jkjQSknw0yZZJNgcuA65Iss5RGyWNpkErED+ezfjuST4GfA3YI8nqJAcDxyf5nySrgD8A3jJgbJLm125VdQRwe1Utp5lg8jEdxyRpNOxZVbcCL6SZM2ZH4JXdhiRptgYdAeHCJB8HPg3cOVlYVZ9a205VdeA0xacMGIukbv2mfb4lyaOBHwE7dxeOpBFy/yT3p6lAfLCqfpPEbsjSmBq0ArElcAfw7J6yAtZagZC0IC1tR0/7W2AF8EDgiG5DkjQiTgauAb4FfDnJTsCtnUYkadYGqkBU1auHFYik8ZXkfsCtVfUz4MvAIzsOSdIIqar3A++fXE/yQ5ruypLG0KwqEEn+oqqOT/IBmhaHNVTVmwaOTNLYqKp7khwCfKLrWCSNtiSfrarnA3d1HYuk2ZltC8TkjdMXDisQSWPvC0neRjPO+73DM1aVs85L6rVd1wFIGsysKhBV9e/t8/LhhiNpjE3O99A7OVRhdyZJa7qk6wAkDWa2XZiWAh+oqv+Z5rXNgZcDd1bVGQPGJ2lMVNUus9kvyXOB9wEbAR+uquNm2O4lNDPY7lNVtn5KYyDJjlX1w96yfieXNDdomJYnwzvYsmXDO9aYmm0Xpn8GjkjyGJoJ335CM/Ps7jQjM50KWHmQNiBJXjVdeVWdvpZ9NgJOBJ4FrAYuSLKiqi6fst0WwJuAbwwvYknz4NPA4wGSfLKqXtzPTuYGabTNtgvTN4GXJXkgMAEsAn4JfLuqrhhifJLGxz49yw8AngFcDMxYgQCeAFxVVVcDJDkT2A+4fMp27wSOB942tGglzYfen33XpzujuUEaYYMO4/oL4PzhhCJpnFXVob3rSbYCPrKO3bYDru1ZXw08ccpxHgfsUFWfbW/SljQ+aobldTE3SCNs0InkJGkmd9B0a1yb6Tql3vtHRju/xAnA4nW9WZIlwBKAHXfcse8gJc2pxya5lea7vlm7TLteVbXlDPuZG6QRZgVC0lAk+Xfu+w/+fsCerHteiNXADj3r2wPX96xvATwaOD/NDXAPB1YkecHUmyWraimwFGBiYmJ9fumUNEeqaqNZ7mpukEbYUCoQSTavqtvXvaWkBezdPct3AT+oqtXr2OcCYPckuwDXAQcAr5h8sap+Djxkcj3J+cDbHGlFWvDMDdIIu98gOyfZN8nltBPLJXlskn8eSmSSxs0PgW9U1Zeq6qvAzUl2XtsOVXUXcAhwDk0e+URVXZbkHUleMNcBSxpN5gZptA3aAnEC8BxgBUBVfSvJUweOStI4OgvYt2f97rZsn+k3b1TVSmDllLIjZ9j26YOFKGlcmBuk0TVQCwRAVV07pejuQY8paSxtXFW/nlxplzfpMB5JkjQHBq1AXJtkX6CSbNIOo/btIcQlafz8pLdrQZL9gJs6jEeSJM2BQbswvZ5mmvntaEZMOBd446BBSRpLrwfOSPLBdn01MO3s1JobyzPdyJeztGzZ8I4lSVpQBp1I7ibgT4YUi6QxVlXfA57UzlCfqrqt65gkSdLwDVSBaIdXOxTYufdYVeUICdIGJsnfA8dX1S3t+oOAP6+qv53vWG6+6KLBf433F3hJkqY1aBemTwOnAP8O3DN4OJLG2POq6q8nV6rqZ0n+CJj3CoQkSZo7g1YgflVV7x9KJJLG3UZJNq2qOwGSbAZs2nFMkiRpyAatQLwvyVE0N0/fOVlYVRevbackpwLPB26sqke3ZQ8GPk7THeoa4GVV9bMB45M0f/4VOC/JaUABrwFO7zYkSZI0bINWIB4DvBL4Q+7rwlTt+tosAz7Imn9cHA6cV1XHJTm8Xf/LAeOTNE+q6vgkq4BnAgHeWVXndByWJEkaskErEPsDj+ydPKofVfXlJDtPKd4PeHq7vBw4HysQ0lipqs8DnwdI8pQkJ1aVQztLkrSADFqB+BawNXDjEGLZtqpuAKiqG5I8bAjHlDSPkuwFHAi8HPg+8KluI5IkOUeMhm3QCsS2wHeSXMCa90DM2TCuSZYASwB23HHHuXobSX1K8tvAATQVh5tp7mVKVf1Bp4FJkqQ5MWgF4qihRNH4cZJFbevDImZo1aiqpcBSgImJiRri+0uane8AXwH+uKquAkjylm5DkqT/bRhzxBxU/ukhDToT9ZeGFQiwAjgIOK59/swQjy1p7ryYpgXii0k+D5xJcxO1JElagO43m52S/Ff7fFuSW3setyW5tY/9PwZ8DdgjyeokB9NUHJ6V5ErgWe26pBFXVWdX1cuBR9EMfvAWYNskJyV5dqfBSZKkoZttC8TmAFW1xWx2rqoDZ3jpGbOMR1LHqup24AzgjHZel5fSDMd8bqeBSZKkoZptBcIOgJJmVFU/BU5uH5I0Eu4CLgHe11O2mGYM+cU9ZY+laUo9gWa4yUnLgKVLl/K6173u3rIVK1aw9957s912291b9md/9mcsXbqUvffem4svbubWXbRoEddffz1HH300xxxzzH0HPXrKMzQD2+8PvBm4pS3bCTgGOA3o7UB+AnANZPF9PUdPPvlklixZQnru91jbOZ3fPk86jGZW396b2Z4GvBrgqKPgBz9oCrfeGt77Xjj7bPhMT8/zo49e8xlgv/1g//3hzW+GW9qT2mmntZ7T+v5DTV6Dqvpf/05rO6ejgPaM2Bp4L3A2a/alP7r3efHidZzTMXDaafClnpM64QS45hp4X89JLV4MT396/x++81nnP9SSry7p+7N34YUXAjAxMcH6Ss3iZqAkq4F/mun1qprxtWGamJioyZOXZivLlw/tWHXQQUM71jAkuaiq1j8zjLldkjp6wGMsHupQhYuHdqQ6aOacPcyhGod7/uA1gPm6Bv0wN8zeMG+izvLhfV79XngNYP5yw2xbIDYCHog3SkqSJEkblNlWIG6oqncMNRJJkiRJI29WozBhy4MkSZK0QZptBcLRkiRJkqQN0KwqEO0IK5IkSZI2MLNtgZAkSZK0AZrtTdSSNLKGMdY7558PvUP1HXYY7LwzvKVnwO2nPQ1e/eo+x0Wf8gyO9e5Y7yM/1rskTWdW80CMCueB0DA4D8TC4zwQgxvXcc69Bv0xN8ye80AsG9qx2iMO7Uheg/nLDXZhkiRJktQ3KxCSJEmS+mYFQpIkSVLfrEBI6lSS5ya5IslVSQ6f5vW3Jrk8yaok5yXZqYs4Jc0vc4M0uqxASOpMko2AE4HnAXsCBybZc8pmlwATVfV7wL8Bx89vlJLmm7lBGm1WICR16QnAVVV1dVX9GjiTZnDTe1XVF6vqjnb168D28xyjpPlnbpBGmBUISV3aDri2Z311WzaTg4HPzWlEkkaBuUEaYU4kJ6lL0w3MPe0g1kn+FJigmRdsuteXAEsAthlWdJK6Ym6QRpgtEJK6tBrYoWd9e+D6qRsleSbwN8ALqurO6Q5UVUuraqKqJraYk1AlzSNzgzTCrEBI6tIFwO5JdkmyCXAAsKJ3gySPA06m+QPhxg5ilDT/zA3SCLMCIakzVXUXcAhwDvBt4BNVdVmSdyR5QbvZPwIPBM5K8s0kK2Y4nKQFwtwgjbaRuwciyTXAbcDdwF1VNdFtRJLmUlWtBFZOKTuyZ/mZ8x6UpM6ZG6TRNXIViNYfVNVNXQchSZIkaU12YZIkSZLUt1GsQBRwbpKL2qHXJEmSJI2IUezC9JSquj7Jw4AvJPlOVX158sXe8Zx33HHHrmKUJEmSNkgj1wJRVde3zzcCZ9NMZ9/7+r3jOT/0oQ/tIkRJkiRpgzVSFYgkmyfZYnIZeDZwabdRSZIkSZo0al2YtgXOTgJNbB+tqs93G5IkSZKkSSNVgaiqq4HHdh2HJEmSpOmNVBcmSZIkSaPNCoQkSZKkvlmBkCRJktQ3KxCSJEmS+mYFQpIkSVLfrEBIkiRJ6psVCEmSJEl9swIhSZIkqW9WICRJkiT1baRmotb8y/LlQz1eHXTQUI8nSZKk0WILhCRJkqS+WYGQJEmS1DcrEJIkSZL6ZgVCkiRJUt+sQEiSJEnqmxUISZIkSX2zAiFJkiSpb1YgJEmSJPXNCoQkSZKkvlmBkCRJktQ3KxCSJEmS+rZx1wFMleS5wPuAjYAPV9Vxc/p+y5cP7Vh10EFDO5a0oVjXdz7JpsDpwN7AzcDLq+qa+Y5T0vwyN0ija6RaIJJsBJwIPA/YEzgwyZ7dRiVprvT5nT8Y+FlV7QacALxrfqOUNN/MDdJoG6kKBPAE4Kqqurqqfg2cCezXcUyS5k4/3/n9gMmmwn8DnpEk8xijpPlnbpBG2Kh1YdoOuLZnfTXwxI5ikdZblg/3/646qIZ6vBHUz3f+3m2q6q4kPwe2AW6alwgldcHcII2wVI3OHyhJXgo8p6pe266/EnhCVR3as80SYEm7ugdwxbwHOrOHYOLa0K/BqJ3/TlX10K6DmEmf3/nL2m1Wt+vfa7e5ecqxzA2jzWswWtfA3DAaRukz0RWvwWhdg75yw6i1QKwGduhZ3x64vneDqloKLJ3PoPqV5MKqmug6ji5t6NdgQz//WVjnd75nm9VJNga2An469UDmhtHmNfAarCdzwwbCazCe12DU7oG4ANg9yS5JNgEOAFZ0HJOkudPPd34FMDnE2UuA/6xRajqVNBfMDdIIG6kWiLYP4yHAOTTDtp1aVZd1HJakOTLTdz7JO4ALq2oFcArwkSRX0fy6eEB3EUuaD+YGabSNVAUCoKpWAiu7jmOWRrKJdJ5t6NdgQz//9Tbdd76qjuxZ/hXw0vmOa8j8XHgNwGuwXswNGwyvwRheg5G6iVqSJEnSaBu1eyAkSZIkjTArEOspyalJbkxyaU/Zu5KsSnJ6T9krkxzWTZTDN8N5PzjJF5Jc2T4/qC1/cZLLknwlyTZt2a5Jzuwq/tlYz3NOkvcnuar9LDy+Ld8jyUVJvpXkyW3Zxkn+I8lvdXNmmgvmhg0nN4D5Qf0zN5gbFmJusAKx/pYBz51cSbIVsG9V/R6wUZLHJNkMWAz8cycRzo1l9Jx363DgvKraHTivXQf4c+BJwOnAK9qyY4Ej5j7MoVpG/+f8PGD39rEEOKktf127zUuAt7VlbwA+UlV3zFnk6sIyzA2TFnpuAPOD+rcMc8Mkc8MCyQ1WINZTVX2ZNceZvgfYJEmAzYDfAG8H3l9Vv+kgxDkxzXkD7Acsb5eXAy9sl+8BNgV+C/hNkv8D3FBVV85HrMOynue8H3B6Nb4ObJ1kEc3nYTPuuxZbA39MkyS1gJgb1rCgcwOYH9Q/c8MazA0LJDeM3ChM46aqbkvySeASmlrlz4F9quod3UY2L7atqhsAquqGJA9ry4+hGXrveuBPgU+wcIbXm+mctwOu7dludVt2Is0XflOaXxSOBP7OscoXPnPDBpcbwPygPpgbzA0LITdYgRiCqjoeOB4gyYeBI5O8Fng2sKqqju0yvvlWVV8AvgCQ5CCaYfj2SPI24GfAYaPSBDdEmaasquqHwNMBkuwGPAL4TpKPAJsAR1TVd+ctSs0rc8OaNtDcAOYHTWFuWJO5YQ1jkRvswjRESR7XLn4XeFVVvQx4dJLdOwxrLv24bWqjfb6x98X2Rp+DaPp0/gPwGuAi4E/mOc5hmumcVwM79Gy3Pc0vKb3+jqY/55uAM4Cj2ocWOHPDBpEbwPyg9WRuMDcwprnBCsRwvZOmmen+NDNnQtOvbyTumJ8DK2i+6LTPn5ny+l8A72v7dG4GFON/PWY65xXAq9oRFZ4E/HyyuRIgydOA69r+nL9Fcx3uZryvhfpnbljTQswNYH7Q+jM3rMncMC65oap8rMcD+BhwA80NLquBg9vyFwJH9Wz3buB/gDO6jnmuzhvYhqb/5pXt84N7tn8E8Nme9ZcClwFfBR7a9fkM+5xpmiFPBL7X/rtP9BwnNE2zD2rXfwe4GFgFPKXr8/Qxd5+XttzcsMByw/qet/lhw36YG8wNCzE3OBO1JEmSpL7ZhUmSJElS36xASJIkSeqbFQhJkiRJfbMCIUmSJKlvViAkSZIk9c0KxJhIsk2Sb7aPHyW5rmd9kz6PcVqSPdaxzRuTDGXCliT7tfF9K8nl7Syba9v+D9txkKd7bVGSlT3HWtGW75Dk48OIVxpH5gZzgzQdc4O5YS45jOsYSnI08IuqeveU8tD8m97TSWBrxrIp8H2aMY2vb9d3qrVMvZ7kWOCmqnrvNK+dAlxcVSe2679XVavmKHxpLJkbzA3SdMwN5oZhswVizCXZLcmlSf6FZoKRRUmWJrkwyWVJjuzZ9r+S7JVk4yS3JDmurZl/LcnD2m2OTfLmnu2PS/LfSa5Ism9bvnmST7b7fqx9r72mhLYVzSQoPwWoqjsnk0CSbZN8qt3vv5M8KcmuwGuBt7e/Puw75XiLaCZkoT3eqp7z/2a7fFrPrys3Jfmbtvzw9n1W9V4PaSEzN5gbpOmYG8wNw2AFYmHYEzilqh5XVdcBh1fVBPBY4FlJ9pxmn62AL1XVY4GvAa+Z4dipqicAbwcmv0SHAj9q9z0OeNzUnarqRuAc4AdJPprkwCSTn7f3A8e3Mb4M+HBVfQ/4MPCPVbVXVf2/KYf8ILA8yX8m+eski6Z5z1dX1V7A/sBNwOlJ/gjYEXgisBew7zRJRlqozA2YG6RpmBswNwzCCsTC8L2quqBn/cAkF9P8svA7NIliql9W1efa5YuAnWc49qem2eb3gTMBqupbNFPN/y9VtRh4FnAhcDiwtH3pmcC/tL8AfBp4UJLNZj49qKqVwK7AKe35XJJkm6nbtcc5C3hDVV0LPBt4HnAJzfXYDfjttb2XtICYG1rmBmkN5oaWuWF2Nu46AA3F7ZMLSXYHDgOeUFW3JPlX4AHT7PPrnuW7mfmzcOc026TfwNomw1VJPgp8m6a5MW18vTGQrP2wVXUzcAZwRpLP0ySkqUnoQ8CZVfXFnliPrapT+o1ZWkDMDfcxN0j3MTfcx9wwC7ZALDxbArcBt7bNdc+Zg/f4L5omRJI8hml+qUiyZZKn9hTtBfygXf4P4I092072g7wN2GK6N0zyjMlfG5JsCewC/HDKNocB959yk9g5wMFJNm+32T7JQ/o8T2khMTeYG6TpmBvMDevNFoiF52LgcuBS4Grgq3PwHh+g6Se4qn2/S4GfT9kmwF8l+RDwS+AX3Ndf8o3ASUleTfMZ/GJb9hngrCQvAt44pT/jPsAHk/yGpuJ7UlVdkmS3nm3eBtwxeXMU8MGq+nCSRwFfb3+puA14BU1fR2lDYm4wN0jTMTeYG9abw7hqvSXZGNi4qn7VNn2eC+xeVXd1HJqkDpkbJE3H3LDw2AKh2XggcF6bEAK8ziQgCXODpOmZGxYYWyAkSZIk9c2bqCVJkiT1zQqEJEmSpL5ZgZAkSZLUNysQkiRJkvpmBUKSJElS36xASJIkSerb/wdXQzGImgvgZAAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Improving-Results">Improving Results<a class="anchor-link" href="#Improving-Results">&#182;</a></h2><p>In this final section, you will choose from the three supervised learning models the <em>best</em> model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (<code>X_train</code> and <code>y_train</code>) by tuning at least one parameter to improve upon the untuned model's F-score.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-3---Choosing-the-Best-Model">Question 3 - Choosing the Best Model<a class="anchor-link" href="#Question-3---Choosing-the-Best-Model">&#182;</a></h3><ul>
<li>Based on the evaluation you performed earlier, in one to two paragraphs, explain to <em>CharityML</em> which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. </li>
</ul>
<p><strong> HINT: </strong> 
Look at the graph at the bottom left from the cell above(the visualization created by <code>vs.evaluate(results, accuracy, fscore)</code>) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:</p>
<ul>
<li>metrics - F score on the testing when 100% of the training data is used, </li>
<li>prediction/training time</li>
<li>the algorithm's suitability for the data.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer: </strong></p>
<p>Out of the 3 models which are as follows:</p>
<ol>
<li>GaussianNB</li>
<li>SVM</li>
<li>Random Forest</li>
</ol>
<p>I believe <strong>Random Forest</strong> is the most appropriate model for identifying individuals that make more than $50,000 because of the following reasons:</p>
<ul>
<li>The Fscore for testing set with 100% training set clearly shows that both the <strong>SVM classifier</strong> as well as the <strong>Random Forest</strong> Classifier performed equally well with a score of around <strong>0.7</strong></li>
<li>But the reason to chose Random Forest over SVM is because the training time taken by SVM is above 200 seconds when 100% of the training set is used, which is significantly greater as compared to Random Forest which less than 1 second.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-4---Describing-the-Model-in-Layman's-Terms">Question 4 - Describing the Model in Layman's Terms<a class="anchor-link" href="#Question-4---Describing-the-Model-in-Layman's-Terms">&#182;</a></h3><ul>
<li>In one to two paragraphs, explain to <em>CharityML</em>, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.</li>
</ul>
<p><strong> HINT: </strong></p>
<p>When explaining your model, if using external resources please include all citations.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer: </strong></p>
<p>The model used is Random Forest. Before explaining what is a Random Forest, we will see what is a Decision Tree.
A decision tree is group of nodes where each node(parent node) splits into many nodes(child node) except for the nodes present at the end. The working of decision trees can be seen analogous to the famous game Twenty Questions. In the game, one player is chosen to be the answerer. That person chooses a subject (object) but does not reveal this to the others. All other players are questioners. They each take turns asking a question which can be answered with a simple "Yes" or "No." trying to guess the subject, the person has chosen.[1]</p>
<p>In our case, the decision is made based on the number of hours per week, education level, marital status and many other features. And based on this it decides whether the individual is earning more than $50,000 or not. A Random Forest is just a bunch of such decision trees. Each tree, on it's own performs poorly. This can be seen in real life as a doctor would have good knowledge about medicine while on the other hand a civil engineer would have good knowledge about constuction of buildings but not medicine. But when both of their knowledge is combined. They can solve problems requiring knowledge of both disciplines.</p>
<p>Similarly, the poorly performing decision trees have limited knowledge and are called weak learners. And when these weak learners' results are combined by averaging, they form a strong learner. Every weak learner votes whether the individual earns more than 50k dollars or not, and based on the majority the final decision is taken. This is done for every individual row of the dataset(i.e. 36177 times to be precise when the 100% of the training set is used). This is how the training of random forest model is done.</p>
<p>For prediction these trained set of decision trees will be used, but the data on which prediction is made is the remaining rows of the data not used for training(i.e. 9045 rows).</p>
<p><strong>Refernces:</strong></p>
<ol>
<li><a href="https://en.wikipedia.org/wiki/Twenty_Questions">https://en.wikipedia.org/wiki/Twenty_Questions</a></li>
</ol>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Implementation:-Model-Tuning">Implementation: Model Tuning<a class="anchor-link" href="#Implementation:-Model-Tuning">&#182;</a></h3><p>Fine tune the chosen model. Use grid search (<code>GridSearchCV</code>) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:</p>
<ul>
<li>Import <a href="http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html"><code>sklearn.grid_search.GridSearchCV</code></a> and <a href="http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html"><code>sklearn.metrics.make_scorer</code></a>.</li>
<li>Initialize the classifier you've chosen and store it in <code>clf</code>.<ul>
<li>Set a <code>random_state</code> if one is available to the same state you set before.</li>
</ul>
</li>
<li>Create a dictionary of parameters you wish to tune for the chosen model.<ul>
<li>Example: <code>parameters = {'parameter' : [list of values]}</code>.</li>
<li><strong>Note:</strong> Avoid tuning the <code>max_features</code> parameter of your learner if that parameter is available!</li>
</ul>
</li>
<li>Use <code>make_scorer</code> to create an <code>fbeta_score</code> scoring object (with $\beta = 0.5$).</li>
<li>Perform grid search on the classifier <code>clf</code> using the <code>'scorer'</code>, and store it in <code>grid_obj</code>.</li>
<li>Fit the grid search object to the training data (<code>X_train</code>, <code>y_train</code>), and store it in <code>grid_fit</code>.</li>
</ul>
<p><strong>Note:</strong> Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># TODO: Import &#39;GridSearchCV&#39;, &#39;make_scorer&#39;, and any other necessary libraries</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">GridSearchCV</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">make_scorer</span><span class="p">,</span><span class="n">fbeta_score</span>
<span class="c1"># TODO: Initialize the classifier</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">()</span>

<span class="c1"># TODO: Create the parameters list you wish to tune, using a dictionary if needed.</span>
<span class="c1"># HINT: parameters = {&#39;parameter_1&#39;: [value1, value2], &#39;parameter_2&#39;: [value1, value2]}</span>
<span class="n">parameters</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;n_estimators&#39;</span><span class="p">:[</span><span class="mi">10</span><span class="p">,</span><span class="mi">20</span><span class="p">,</span><span class="mi">30</span><span class="p">,</span><span class="mi">50</span><span class="p">],</span><span class="s1">&#39;criterion&#39;</span><span class="p">:(</span><span class="s1">&#39;entropy&#39;</span><span class="p">,</span><span class="s1">&#39;gini&#39;</span><span class="p">),</span><span class="s1">&#39;max_features&#39;</span><span class="p">:[</span><span class="s1">&#39;auto&#39;</span><span class="p">,</span><span class="s1">&#39;log2&#39;</span><span class="p">,</span><span class="kc">None</span><span class="p">]}</span>

<span class="c1"># TODO: Make an fbeta_score scoring object using make_scorer()</span>
<span class="n">scorer</span> <span class="o">=</span> <span class="n">make_scorer</span><span class="p">(</span><span class="n">fbeta_score</span><span class="p">,</span><span class="n">beta</span><span class="o">=</span><span class="mf">0.5</span><span class="p">)</span>

<span class="c1"># TODO: Perform grid search on the classifier using &#39;scorer&#39; as the scoring method using GridSearchCV()</span>
<span class="n">grid_obj</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">clf</span><span class="p">,</span><span class="n">parameters</span><span class="p">,</span><span class="n">scoring</span><span class="o">=</span><span class="n">scorer</span><span class="p">)</span>

<span class="c1"># TODO: Fit the grid search object to the training data and find the optimal parameters using fit()</span>
<span class="n">grid_fit</span> <span class="o">=</span> <span class="n">grid_obj</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>

<span class="c1"># Get the estimator</span>
<span class="n">best_clf</span> <span class="o">=</span> <span class="n">grid_fit</span><span class="o">.</span><span class="n">best_estimator_</span>

<span class="c1"># Make predictions using the unoptimized and model</span>
<span class="n">predictions</span> <span class="o">=</span> <span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">))</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">best_predictions</span> <span class="o">=</span> <span class="n">best_clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>

<span class="c1"># Report the before-and-afterscores</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Unoptimized model</span><span class="se">\n</span><span class="s2">------&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy score on testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;F-score on testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">predictions</span><span class="p">,</span> <span class="n">beta</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="se">\n</span><span class="s2">Optimized Model</span><span class="se">\n</span><span class="s2">------&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Final accuracy score on the testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">best_predictions</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Final F-score on the testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">best_predictions</span><span class="p">,</span> <span class="n">beta</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">)))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Unoptimized model
------
Accuracy score on testing data: 0.8377
F-score on testing data: 0.6715

Optimized Model
------
Final accuracy score on the testing data: 0.8463
Final F-score on the testing data: 0.6888
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-5---Final-Model-Evaluation">Question 5 - Final Model Evaluation<a class="anchor-link" href="#Question-5---Final-Model-Evaluation">&#182;</a></h3><ul>
<li>What is your optimized model's accuracy and F-score on the testing data? </li>
<li>Are these scores better or worse than the unoptimized model? </li>
<li>How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in <strong>Question 1</strong>?_  </li>
</ul>
<p><strong>Note:</strong> Fill in the table below with your results, and then provide discussion in the <strong>Answer</strong> box.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Results:">Results:<a class="anchor-link" href="#Results:">&#182;</a></h4><table>
<thead><tr>
<th style="text-align:center">Metric</th>
<th style="text-align:center">Unoptimized Model</th>
<th style="text-align:center">Optimized Model</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">Accuracy Score</td>
<td style="text-align:center">0.8377</td>
<td style="text-align:center">0.8463</td>
</tr>
<tr>
<td style="text-align:center">F-score</td>
<td style="text-align:center">0.6715</td>
<td style="text-align:center">0.6888</td>
</tr>
</tbody>
</table>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer: </strong></p>
<p>Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2971]
The optmized model's score are definetely better than the unoptimized RF model. The parameter set in the optimized model are as follows:</p>
<ul>
<li>criterion = 'entropy'</li>
<li>max_features = None</li>
<li>n_estimators = 50</li>
</ul>
<p>Now, compared to the Naive Predictor below is the table showing the difference</p>
<table>
<thead><tr>
<th style="text-align:center">Metric</th>
<th style="text-align:center">Naive Predictor</th>
<th style="text-align:center">Optimized Model</th>
<th style="text-align:center">Difference</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">Accuracy Score</td>
<td style="text-align:center">0.2478</td>
<td style="text-align:center">0.8463</td>
<td style="text-align:center">0.5985</td>
</tr>
<tr>
<td style="text-align:center">F-score</td>
<td style="text-align:center">0.2917</td>
<td style="text-align:center">0.6888</td>
<td style="text-align:center">0.3971</td>
</tr>
</tbody>
</table>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Feature-Importance">Feature Importance<a class="anchor-link" href="#Feature-Importance">&#182;</a></h2><p>An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.</p>
<p>Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a <code>feature_importance_</code> attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-6---Feature-Relevance-Observation">Question 6 - Feature Relevance Observation<a class="anchor-link" href="#Question-6---Feature-Relevance-Observation">&#182;</a></h3><p>When <strong>Exploring the Data</strong>, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer:</strong></p>
<p>The five most important features according to me out of the 13 features in order are:</p>
<ol>
<li>hours-per-week</li>
<li>capital-gain</li>
<li>age</li>
<li>occupation</li>
<li>education-level</li>
</ol>
<p>Because the hours-per-week is directly related to the income as income is on hourly basis. After that the capital gains affect as you are not working constantly at it but once you work on it, it keeps on generating some amount of income regularly. So larger the capital gain, the higher would be your net income. Age is also a factor because a higher income also depends the amount of experience a person has. Occupation beacause some occupation may be simply highly rewarding than others. Then comes the education level as generally people with higher degrees are also paid higher.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Implementation---Extracting-Feature-Importance">Implementation - Extracting Feature Importance<a class="anchor-link" href="#Implementation---Extracting-Feature-Importance">&#182;</a></h3><p>Choose a <code>scikit-learn</code> supervised learning algorithm that has a <code>feature_importance_</code> attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.</p>
<p>In the code cell below, you will need to implement the following:</p>
<ul>
<li>Import a supervised learning model from sklearn if it is different from the three used earlier.</li>
<li>Train the supervised model on the entire training set.</li>
<li>Extract the feature importances using <code>'.feature_importances_'</code>.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># TODO: Import a supervised learning model that has &#39;feature_importances_&#39;</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>

<span class="c1"># TODO: Train the supervised model on the training set using .fit(X_train, y_train)</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span><span class="n">max_features</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span><span class="n">criterion</span><span class="o">=</span><span class="s1">&#39;entropy&#39;</span><span class="p">)</span>

<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>

<span class="c1"># TODO: Extract the feature importances using .feature_importances_ </span>
<span class="n">importances</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">feature_importances_</span>

<span class="c1"># Plot</span>
<span class="n">vs</span><span class="o">.</span><span class="n">feature_plot</span><span class="p">(</span><span class="n">importances</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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POOMM4DQ4jZ+/HhOO+200mUrVqxYr3zHjh15++23Wbp0KStXrmT77bdnr732YtasWYwfP55f//rXALz00ktMmzat9HnfffcdixcvXqeuCRMm8MwzzwBw1llncfXVV6+znmbNmgHQtm1b5syZUzrOUGRzZA8/XG11+3nnVVvdsr6CJIBmVgO4BzgamAtMMrOR7l565nT3KxPlfwW0K0RsIrLpuTt/+ctf6N69+zrzhwwZwoIFC5g8eTK1atWiqKgo533hatasyZo1a0qns8tst912AKxZs4aGDRuul4Bmq1evHnvvvTeDBw+mffv2AHTq1IlRo0Yxf/58WrVqVVrfhAkTqFu3buVfNFC7du3SxzVq1GDVqlUbVI+ISGUVagxgR2CWu3/o7j8Aw4CTyih/JvB4QSITkU2ue/fu/N///R8rV64E4IMPPmDp0qUsWrSInXfemVq1ajF69Gg+/vhjAOrXr79OS9uee+7JtGnTWLFiBYsWLeLll1/OuZ4ddtiBFi1a8NRTTwEh8XznnXdylu3SpQsDBw6kc+fOAHTu3Jm77rqLTp06ld6f7phjjuHuu+8ufU6uxLJTp048/fTTAAwbNqxC2yP79YmIVLVCdQHvDnyamJ4LHJSroJntCbQAXsmzvC/QF2CPPfao2ihFUqYit20phAsuuIA5c+bQvn370tuwPPPMM5x99tmceOKJFBcX07ZtW/bdd18AGjVqRJcuXdh///059thjue222zj99NNp06YNLVu2pF27/B0IQ4cO5eKLL+bmm29m5cqV9O7dm5/85CfrlevSpQt33XVXaQLYvn175s6dywUXXFBaZtCgQVx66aW0adOGVatW0bVr1/XG+Q0cOJBzzjmHO+64g+OPP54GDRqUuz369u3LscceS9OmTRk9enSFtqGISGWYe/V/AJjZaUB3d78gTp8LdHT3X+Uoey3QLNeybMXFxV5SUlLl8YpsraZPn85+++23qcNIlWXLllG3bl3MjGHDhvH444+XXs0s5dMxu3nRGMDNn5lNdvfi8soVqgVwLtA8Md0MmJenbG/g0mqPSESkACZPnky/fv1wdxo2bMjgwYM3dUgiIgVLACcBLc2sBfAZIck7K7uQmbUCdgQmFCguEZFqdeihh+YdZygisqkU5CIQd18F9ANeAKYDT7r7e2Y2wMx6JoqeCQzzQvRLi4iIiKRUwe4D6O6jgFFZ8/pnTd9UqHhERERE0ko/BSciIiKSMkoARURERFJGvwUskmJVfUuHitzG4YsvvuCKK65g0qRJ1K5dm6KiIgYOHMg+++xTpbEkdevWjdtvv53i4vx3Rhg4cCB9+/alXr16ABx33HE89thjNGy4cb9IWVRURP369alRowYA9957LwcffHCl6/njH//IDTfcsFGx5NOuXTv+9re/0bZtW1atWkWDBg24//77OeeccwA48MADefDBB0t/FSVbSUkJjzzyCIMGDcq7jjlz5nDCCScwderU9ZYNGTKEY445ht12261qXpCIlEstgCJSMO7OKaecQrdu3Zg9ezbTpk3jj3/84wb9vm9VGzhwIMuWLSudHjVq1EYnfxmjR49mypQpTJkyZYOSPwgJYGVV9KflDj74YMaPHw/AO++8Q6tWrUqnly5dyocffpjzZtkZxcXFZSZ/5RkyZAjz5uW7M5iIVAclgCJSMKNHj6ZWrVpcdNFFpfPatm3LoYceypgxYzjhhBNK5/fr148hQ4YAoRXthhtuoHPnzhQXF/PWW2/RvXt3fvSjH5X+8kZZz0+6+OKLKS4upnXr1tx4441A+EWPefPmcfjhh3P44YeXrvOrr77i2muv5d577y19/k033cQdd9wBwG233UaHDh1o06ZNaV0Vle+5J598MgceeCCtW7fmgQceAOC6667j+++/p23btpx99tnMmTOH/fffv/Q5t99+OzfddBMQWjtvuOEGDjvsMO666y4WLFjAqaeeSocOHejQoQPjxo1bL5YuXbqUJnzjx4/noosuKv1Zu4kTJ9K+fXtq1KjB0qVL+cUvfkGHDh1o165d6Q2tk9t+wYIFHH300bRv355f/vKX7Lnnnnz11VcArF69mgsvvJDWrVtzzDHH8P333zN8+HBKSko4++yzadu2Ld9//32ltqOIbBglgCJSMFOnTuXAAw/coOc2b96cCRMmcOihh9KnTx+GDx/OG2+8Qf/+/ct/csItt9xCSUkJ7777Lq+++irvvvsul112GbvtthujR49e76fXevfuzRNPPFE6/eSTT3Laaafx4osvMnPmTCZOnMiUKVOYPHkyY8eOzbnOww8/nLZt23LQQeEXMMt67uDBg5k8eTIlJSUMGjSIhQsXcuutt1K3bl2mTJnC0KFDy32N3377La+++iq//vWvufzyy7nyyiuZNGkSTz/99Do/ZZeRbAEcP348Xbt2pXbt2ixevJjx48fTpUuX0m13xBFHMGnSJEaPHs0111zD0qVL16nr97//PUcccQRvvfUWp5xyCp988knpspkzZ3LppZfy3nvv0bBhQ55++ml69epFcXExQ4cOZcqUKdStW7fc1yciG09jAEVki9CzZ7hl6AEHHMCSJUuoX78+9evXp06dOnz77bcVrufJJ5/kgQceYNWqVXz++edMmzaNNm3a5C3frl075s+fz7x581iwYAE77rgje+yxB4MGDeLFF18s/d3hJUuWMHPmTLp27bpeHaNHj6buLYAZAAAgAElEQVRx48al0y+++GLe5w4aNIgRI0YA8OmnnzJz5kwaNWpU4dcHcMYZZ5Q+fumll5g2bVrp9HfffcfixYupX79+6byioiJ++OEHvvjiC2bMmEGrVq3o0KEDb775JuPHj+dXv/pVadwjR47k9ttvB2D58uXrJHgAr7/+emn8PXr0YMcddyxd1qJFC9q2bQuEcYVz5syp1OsSkaqjBFBECqZ169YMHz4857KaNWuyZs2a0unly5evs7x27doAbLPNNqWPM9OrVq0q9/kAH330EbfffjuTJk1ixx13pE+fPjnLZevVqxfDhw/niy++oHfv3kAYz3j99dfzy1/+stznZ8v33DFjxvDSSy8xYcIE6tWrR7du3XLGV95r3W677Uofr1mzhgkTJpTbsta5c2eGDx9O06ZNMTM6derEuHHjmDhxIp06dSqN++mnn6ZVq1brPDc5hrOs+/gn91uNGjXU3SuyCakLWEQK5ogjjmDFihU8+OCDpfMmTZrEq6++yp577sm0adNYsWIFixYt4uWXX65U3RV5/nfffcd2221HgwYN+PLLL/n3v/9duqx+/fosXrw4Z929e/dm2LBhDB8+nF69egHQvXt3Bg8ezJIlSwD47LPPmD9/foVizffcRYsWseOOO1KvXj1mzJjBG2+8UfqcWrVqsXLlSgB22WUX5s+fz8KFC1mxYgXPPvts3nUdc8wx3H333aXTmbF92bp06cKdd95J586dgZAQPvLII+y6666lF8N0796dv/zlL6VJ3ttvv71ePYcccghPPvkkEFoMv/nmm3K3R1nbXkSqh1oARVKsIrdtqUpmxogRI7jiiiu49dZbqVOnTultYJo3b87pp59OmzZtaNmyZWn3aEVV5Pk/+clPaNeuHa1bt2avvfYqHdsG0LdvX4499liaNm263jjA1q1bs3jxYnbffXeaNm0KhMRq+vTppQnT9ttvz6OPPsrOO+9cbqz5ntujRw/uu+8+2rRpQ6tWrUpb3jLxtWnThvbt2zN06FD69+/PQQcdRIsWLdh3333zrmvQoEFceumltGnThlWrVtG1a9fSC2eSunTpwpVXXlkaU9OmTVm9evU6Vy3/7ne/44orrqBNmza4O0VFReslnzfeeCNnnnkmTzzxBIcddhhNmzalfv36pcluLn369OGiiy6ibt26FWqtFJGNZ1vyz+4WFxd7SUnJpg5DZIsxffp09ttvv00dhmzFVqxYQY0aNahZsyYTJkzg4osvztvqWBE6ZjcvVX3v0KRCfyHdWpnZZHfPf9PTSC2AIiJSZT755BNOP/101qxZw7bbbrtOd7+IbD6UAIqISJVp2bJlzrGBIrJ50UUgIimzJQ/7kHTRsSpSfZQAiqRInTp1WLhwoT5YZbPn7ixcuJA6deps6lBEtkrqAhZJkWbNmjF37lwWLFiwqUMRKVedOnVo1qzZpg5DZKukBFAkRWrVqkWLFi02dRgiIrKJqQtYREREJGWUAIqIiIikjBJAERERkZRRAigiIiKSMkoARURERFJGCaCIiIhIyigBFBEREUkZJYAiIiIiKaMEUERERCRllACKiIiIpIwSQBEREZGU0W8Bi4hItbGHH662uv2886qtbpGtXcFaAM2sh5m9b2azzOy6PGVON7NpZvaemT1WqNhERERE0qQgLYBmVgO4BzgamAtMMrOR7j4tUaYlcD3Qxd2/MbOdCxGbiIiISNoUqgWwIzDL3T909x+AYcBJWWUuBO5x928A3H1+gWITERERSZVCJYC7A58mpufGeUn7APuY2Tgze8PMeuSqyMz6mlmJmZUsWLCgmsIVERER2XoVKgG0HPM8a7om0BLoBpwJPGRmDdd7kvsD7l7s7sVNmjSp8kBFREREtnaFSgDnAs0T082AeTnK/NPdV7r7R8D7hIRQRERERKpQoRLASUBLM2thZtsCvYGRWWWeAQ4HMLPGhC7hDwsUn4iIiEhqFCQBdPdVQD/gBWA68KS7v2dmA8ysZyz2ArDQzKYBo4Fr3H1hIeITERERSZOC3Qja3UcBo7Lm9U88duCq+CciIiIi1UQ/BSciIiKSMkoARURERFJGCaCIiIhIyigBFBEREUkZJYAiIiIiKaMEUERERCRllACKiIiIpIwSQBEREZGUUQIoIiIikjJKAEVERERSRgmgiIiISMooARQRERFJGSWAIiIiIimjBFBEREQkZZQAioiIiKSMEkARERGRlFECKCIiIpIySgBFREREUkYJoIiIiEjKKAEUERERSRklgCIiIiIpowRQREREJGWUAIqIiIikjBJAERERkZRRAigiIiKSMkoARURERFJGCaCIiIhIyigBFBEREUkZJYAiIiIiKVOwBNDMepjZ+2Y2y8yuy7G8j5ktMLMp8e+CQsUmIiIikiY1C7ESM6sB3AMcDcwFJpnZSHefllX0CXfvV4iYRERERNKqUC2AHYFZ7v6hu/8ADANOKtC6RURERCShIC2AwO7Ap4npucBBOcqdamZdgQ+AK9390xxlRGQrZQ8/XG11+3nnVVvdIiJbmkK1AFqOeZ41/S+gyN3bAC8BOT8JzKyvmZWYWcmCBQuqOEwRERGRrV+hEsC5QPPEdDNgXrKAuy909xVx8kHgwFwVufsD7l7s7sVNmjSplmBFREREtmaFSgAnAS3NrIWZbQv0BkYmC5hZ08RkT2B6gWITERERSZWCjAF091Vm1g94AagBDHb398xsAFDi7iOBy8ysJ7AK+BroU4jYRERERNKmUBeB4O6jgFFZ8/onHl8PXF+oeERERETSSr8EIiIiIpIySgBFREREUkYJoIiIiEjKKAEUERERSRklgCIiIiIpowRQREREJGWUAIqIiIikjBJAERERkZRRAigiIiKSMkoARURERFJGCaCIiIhIyigBFBEREUkZJYAiIiIiKaMEUERERCRllACKiIiIpIwSQBEREZGUUQIoIiIikjJKAEVERERSRgmgiIiISMooARQRERFJGSWAIiIiIimjBFBEREQkZZQAioiIiKSMEkARERGRlFECKCIiIpIyFU4Azey0PPN7VV04IiIiIlLdKtMC+Nc88x+oikBEREREpDBqllfAzPaKD7cxsxaAJRbvBSyvjsBEREREpHqUmwACswAnJH6zs5Z9AdxUxTGJiIiISDUqNwF0920AzOxVdz+s+kMSERERkepU4TGAG5v8mVkPM3vfzGaZ2XVllOtlZm5mxRuzPhERERHJrSJdwADE8X+3AG2B7ZPL3H2Pcp5bA7gHOBqYC0wys5Hu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QRWGKCIiIpIOhUoALce8nH2JZnYOUAzclmu5uz/g7sXuXtykSZMqDFFEREQkHQoyBpDQ4tc8Md0MmJddyMyOAn4DHObuKwoUm4iIiEiqFKoFcBLQ0sxamNm2QG9gZLKAmbUD7gd6uvv8AsUlIiIikjoFSQDdfRXQD3gBmA486e7vmdkAM+sZi90GbA88ZWZTzGxknupEREREZCMUqgsYdx8FjMqa1z/x+KhCxSIiIiKSZgVLAEW2NPZwrmuXtj66t6OISProp+BEREREUkYJoIiIiEjKKAEUERERSRklgCIiIiIpowRQREREJGWUAIqIiIikjBJAERERkZRRAigiIiKSMkoARURERFJGCaCIiIhIyigBFBEREUkZJYAiIiIiKaMEUERERCRllACKiIiIpIwSQBEREZGUUQIoIiIikjJKAEVERERSRgmgiIiISMooARQRERFJGSWAIiIiIimjBFBEREQkZZQAioiIiKSMEkARERGRlFECKCIiIpIySgBFREREUkYJoIiIiEjK1NzUAYiIFII9bJs6hILx83xThyAimzm1AIqIiIikTMESQDPrYWbvm9ksM7sux/KuZvaWma0ys16FiktEREQkbQrSBWxmNYB7gKOBucAkMxvp7tMSxT4B+gBXFyImERHZsqWlW19d+lIdCjUGsCMwy90/BDCzYcBJQGkC6O5z4rI1BYpJREREJJUK1QW8O/BpYnpunFdpZtbXzErMrGTBggVVEpyIiIhImhQqAczVTr9Bbdru/oC7F7t7cZMmTTYyLBEREZH0KVQX8FygeWK6GTCvQOsWERGRzVxaxnTC5jGus1AtgJOAlmbWwsy2BXoDIwu0bhERERFJKEgC6O6rgH7AC8B04El3f8/MBphZTwAz62Bmc4HTgPvN7L1CxCYiIiKSNgX7JRB3HwWMyprXP/F4EqFrWERERESqkX4JRERERCRllACKiIiIpIwSQBEREZGUUQIoIiIikjJKAEVERERSRgmgiIiISMooARQRERFJGSWAIiIiIimjBFBEREQkZZQAioiIiKSMEkARERGRlFECKCIiIpIySgBFREREUkYJoIiIiEjKKAEUERERSRklgCIiIiIpowRQREREJGWUAIqIiIikjBJAERERkZRRAigiIiKSMkoARURERFJGCaCIiIhIyigBFBEREUkZJYAiIiIiKaMEUERE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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-7---Extracting-Feature-Importance">Question 7 - Extracting Feature Importance<a class="anchor-link" href="#Question-7---Extracting-Feature-Importance">&#182;</a></h3><p>Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.</p>
<ul>
<li>How do these five features compare to the five features you discussed in <strong>Question 6</strong>?</li>
<li>If you were close to the same answer, how does this visualization confirm your thoughts? </li>
<li>If you were not close, why do you think these features are more relevant?</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer:</strong></p>
<p>From the features obtained using the <em>.features<em>importance</em></em> and the features i believed were important, three features seem to be common namely:</p>
<ul>
<li><p><strong>capital-gain</strong></p>
<p>As i mentioned in the previous answer, captial-gains is the passive income the person is earning from his investments or properties which obviously adds up to his active income, i.e. the salary he/she might be earning from their job. So it makes it an important factor in determining whether or not the person has captial gains or not.</p>
</li>
<li><p><strong>age</strong></p>
<p>Because a higher income also depends the amount of experience a person has.</p>
</li>
<li><p><strong>hours-per-week</strong></p>
<p>hours-per-week is directly related to the income as income is on hourly basis</p>
</li>
</ul>
<p>But the order of the features, which i believed, has been completely different from the ones obtained using <em>.features<em>importance</em></em>.
And the other two important features are:</p>
<ul>
<li><p><strong>marital_status</strong></p>
<p>It is has been observed from the visualization that people with the marital status "<em>Married-civ-spouse</em>" are the most important ones in determining that there income is greater than $50,000</p>
</li>
<li><p><strong>education_num</strong></p>
<p>Instead of education_num, which represents the number of years one has spent for education i suggested education-level but from a data analyst's education number makes more sense numerically, as for education-level we might have to assign weights according to the order in which the level is increased. For example, a person with masters should be assigned larger weight than a person with bachelors and so on.</p>
</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Feature-Selection">Feature Selection<a class="anchor-link" href="#Feature-Selection">&#182;</a></h3><p>How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of <strong>all</strong> features present in the data. This hints that we can attempt to <em>reduce the feature space</em> and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set <em>with only the top five important features</em>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Import functionality for cloning a model</span>
<span class="kn">from</span> <span class="nn">sklearn.base</span> <span class="k">import</span> <span class="n">clone</span>

<span class="c1"># Reduce the feature space</span>
<span class="n">X_train_reduced</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="n">X_train</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">values</span><span class="p">[(</span><span class="n">np</span><span class="o">.</span><span class="n">argsort</span><span class="p">(</span><span class="n">importances</span><span class="p">)[::</span><span class="o">-</span><span class="mi">1</span><span class="p">])[:</span><span class="mi">5</span><span class="p">]]]</span>
<span class="n">X_test_reduced</span> <span class="o">=</span> <span class="n">X_test</span><span class="p">[</span><span class="n">X_test</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">values</span><span class="p">[(</span><span class="n">np</span><span class="o">.</span><span class="n">argsort</span><span class="p">(</span><span class="n">importances</span><span class="p">)[::</span><span class="o">-</span><span class="mi">1</span><span class="p">])[:</span><span class="mi">5</span><span class="p">]]]</span>

<span class="c1"># Train on the &quot;best&quot; model found from grid search earlier</span>
<span class="n">clf</span> <span class="o">=</span> <span class="p">(</span><span class="n">clone</span><span class="p">(</span><span class="n">best_clf</span><span class="p">))</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_reduced</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

<span class="c1"># Make new predictions</span>
<span class="n">reduced_predictions</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test_reduced</span><span class="p">)</span>

<span class="c1"># Report scores from the final model using both versions of data</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Final Model trained on full data</span><span class="se">\n</span><span class="s2">------&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy on testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">best_predictions</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;F-score on testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">best_predictions</span><span class="p">,</span> <span class="n">beta</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="se">\n</span><span class="s2">Final Model trained on reduced data</span><span class="se">\n</span><span class="s2">------&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Accuracy on testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">reduced_predictions</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;F-score on testing data: </span><span class="si">{:.4f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">fbeta_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">reduced_predictions</span><span class="p">,</span> <span class="n">beta</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">)))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Final Model trained on full data
------
Accuracy on testing data: 0.8469
F-score on testing data: 0.6910

Final Model trained on reduced data
------
Accuracy on testing data: 0.8362
F-score on testing data: 0.6695
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Question-8---Effects-of-Feature-Selection">Question 8 - Effects of Feature Selection<a class="anchor-link" href="#Question-8---Effects-of-Feature-Selection">&#182;</a></h3><ul>
<li>How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?</li>
<li>If training time was a factor, would you consider using the reduced data as your training set?</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Answer:</strong></p>
<p>The final model's accuracy is much affected, it reduced just by 1.26% while the F-score reduced by an absolute value of 0.0215. 
As evident from the numbers the accuracy and F-score is slightly affected but on the other hand the data has been reduced greatly giving a better training time with almost similar prediction results. So yes if training time would have been a factor, it would be wise to use reduced data for training with minimal loss on the accuracy and F-score front.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<blockquote><p><strong>Note</strong>: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to<br>
<strong>File -&gt; Download as -&gt; HTML (.html)</strong>. Include the finished document along with this notebook as your submission.</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Before-You-Submit">Before You Submit<a class="anchor-link" href="#Before-You-Submit">&#182;</a></h2><p>You will also need run the following in order to convert the Jupyter notebook into HTML, so that your submission will include both files.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="o">!!</span>jupyter nbconvert *.ipynb
</pre></div>

</div>
</div>
</div>

</div>
    </div>
  </div>
</body>

 


</html>

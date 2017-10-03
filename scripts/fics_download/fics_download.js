var request = require('sync-request');
var querystring = require('querystring')
const cheerio = require('cheerio')
const child_process = require('child_process')

const URL="http://ficsgames.org/cgi-bin/download.cgi"

const START_YEAR=2016
const END_YEAR=2016
const START_MONTH=1
const END_MONTH=12

function downloadMonth(year, month) {
   console.log("Downloading " + year + "." + month)
   var postData = querystring.stringify({
      'gametype' : '4',
      'player': '',
      'year': year,
      'month' : month,
      'movetimes' : 1,
      'download': 'Download'
  })
  
  var res = request('POST', URL, {
    body: postData
  })

  dom = cheerio.load(res.getBody('utf8'))
  child_process.execSync("wget http://ficsgames.org/"+dom(".messagetext a").attr("href"))
}

for (var year=START_YEAR;year <= END_YEAR; year++) {
  for (var month=START_MONTH; month <= END_MONTH; month++) {
    downloadMonth(year, month)
  }
}


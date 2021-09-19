
const { response } = require('express')
const express = require('express')
const fs = require('fs')
const path = require('path')
const multer = require('multer')

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/')
  },
  filename: function (req, file, cb) {
    const name = req.headers.name
    cb(null, `${name}-${Date.now()}${path.extname(file.originalname)}`)
  }
})

const upload = multer({ storage: storage });

const app = express()
const port = 3000

let images = fs.readdirSync(path.join(__dirname, 'uploads')).filter((file) => file.endsWith('.jpg'))

app.use('/', express.static(path.join(__dirname, '/client')))
app.use('/images', express.static(path.join(__dirname, '/uploads')))
app.use(express.json()) //limit available here

app.get('/images', (req, res) => {
  res.json(images)
})

app.post('/send', upload.any(), (req, res) => {
  console.log(req.files)
  console.log(req.headers)
  res.json({msg: 'images here'})
  refresh()
})

app.listen(port, () => {
  console.log(`lol finally at http://localhost:${port}`)
})

function refresh() {
  images = fs.readdirSync(path.join(__dirname, 'uploads')).filter((file) => file.endsWith('.jpg'))
}

console.log(images)

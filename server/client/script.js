let images
const imageOnWebsite = document.querySelector('#images')

async function init() {
  images = await getImages()
  displayImages()
}

async function getImages() {
  const response = await fetch('/images')
  const json = await response.json()
  return json
}

function displayImages() {
  for (let i = 0; i < images.length; i++) {
    const src = `images/${images[i]}`
    console.log(images[i])

    const el = document.createElement('img')
    el.src = src
    imageOnWebsite.appendChild(el)
  }
}

init()
<script>
import VueDrawingCanvas from 'vue-drawing-canvas'

export default {
  name: 'CanvasComponent',
  components: {
    VueDrawingCanvas,
  },
}
</script>

<template>
  <vue-drawing-canvas ref="VueCanvasDrawing" />
</template>

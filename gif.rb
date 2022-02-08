class GIF
  attr_accessor :width
  attr_accessor :height
  attr_accessor :source_color_depth
  attr_accessor :background_color_index
  attr_accessor :aspect_ratio
  attr_accessor :palette
  attr_accessor :loop_count
  attr_accessor :graphics
  attr_accessor :comments

  def initialize
    @width = 0
    @height = 0
    @source_color_depth = 1
    @background_color_index = 0
    @aspect_ratio = nil
    @palette = nil
    @loop_count = nil
    @comments = []
    @graphics = []
  end

  class Palette
    attr_reader :depth
    attr_accessor :sorted

    def initialize(depth)
      @depth = depth
      @colors = Array.new(1 << depth, 0)
      @sorted = false
    end

    def size
      @colors.size
    end

    def [](i)
      @colors[i]
    end

    def []=(i, color)
      @colors[i] = color
    end
  end

  class Graphic
    # Disposal methods
    NONE = 0
    KEEP = 1
    FILL = 2
    RESTORE = 3

    attr_accessor :disposal_method
    attr_accessor :delay_time
    attr_accessor :user_input
    attr_accessor :transparent_color_index

    def initialize(graphic = nil)
      if graphic.nil?
        @disposal_method = NONE
        @delay_time = nil
        @user_input = false
        @transparent_color_index = nil
      else
        @disposal_method = graphic.disposal_method
        @delay_time = graphic.delay_time
        @user_input = graphic.user_input
        @transparent_color_index = graphic.transparent_color_index
      end
    end
  end

  class Text < Graphic
    attr_accessor :x
    attr_accessor :y
    attr_accessor :width
    attr_accessor :height
    attr_accessor :cell_width
    attr_accessor :cell_height
    attr_accessor :foreground_color_index
    attr_accessor :background_color_index
    attr_accessor :data

    def initialize(graphic = nil)
      super(graphic)
      @x = 0
      @y = 0
      @width = 0
      @height = 0
      @cell_width = 0
      @cell_height = 0
      @foreground_color_index = 0
      @background_color_index = 1
      @data = [].pack('C*')
    end
  end

  class Image < Graphic
    attr_accessor :x
    attr_accessor :y
    attr_accessor :width
    attr_accessor :height
    attr_accessor :palette
    attr_accessor :interlaced
    attr_accessor :data

    def initialize(graphic)
      super(graphic)
      @x = 0
      @y = 0
      @width = 0
      @height = 0
      @palette = nil
      @data = [].pack('C*')
    end

    def color_index(x, y)
      @data.getbyte(interlace(y) * @width + x)
    end

    def set_color_index(x, y, index)
      @data.setbyte(interlace(y) * @width + x, index)
    end

    def interlace(y)
      return y unless @interlaced
      i = y / @height
      y %= @height
      for step, start in [[8, 0], [8, 4], [4, 2], [2, 1]]
        size = (@height - start + step - 1) / step
        return i * @height + step * y + start if y < size
        y -= size
      end
      nil
    end
  end

  class Stream
    def self.open_file(*args, &block)
      open(File.open(*args), &block)
    end

    def self.open(io)
      stream = new(io)
      if block_given?
        begin
          yield stream
        ensure
          stream.close
        end
      else
        stream
      end
    end

    def initialize(io)
      @io = io
    end

    def close
      @io.close
    end
  end

  class Reader < Stream
    def self.read_file(path)
      open_file(path, 'rb') { |reader| reader.read }
    end

    def read
      gif = GIF.new
      read_header
      read_body(gif)
      gif
    end

    def read_header
      version = read_bytes(6)
      raise IOError, "unsupported version #{version}" unless version == 'GIF89a'
    end

    def read_body(gif)
      read_screen_descriptor(gif)
      graphic = nil
      loop do
        sep = read_byte.chr
        case sep
        when '!'
          label = read_byte
          case label
          when 0x01
            graphic = read_plain_text_extension(graphic)
            gif.graphics << graphic
            graphic = nil
          when 0xF9
            raise IOError, "multiple open graphics" unless graphic.nil?
            graphic = Graphic.new
            read_graphic_control_extension(graphic)
          when 0xFE
            gif.comments << read_remaining_blocks.join
          when 0xFF
            read_app_extension(gif)
          else
            graphic = nil if label.between?(0x00, 0x7F)
            read_remaining_blocks
          end
        when ','
          graphic = read_image(graphic)
          gif.graphics << graphic
          graphic = nil
        when ';'
          break
        else
          raise IOError, "unsupported block separator - #{sep}"
        end
      end
      raise IOError, "unclosed graphic" unless graphic.nil?
    end

    def read_screen_descriptor(gif)
      gif.width, gif.height, bits, gif.background_color_index, aspect_byte = read_bytes(7).unpack('vvCCC')
      gif.source_color_depth = ((bits >> 4) & 0x7) + 1
      gif.aspect_ratio = aspect_byte == 0 ? nil : (aspect_byte + 15) / 64.0
      if (bits & 0x80) == 0x80
        gif.palette = read_palette((bits & 0x7) + 1)
        gif.palette.sorted = (bits & 0x08) == 0x08
      end
    end

    def read_palette(depth)
      palette = Palette.new(depth)
      for i in 0 ... palette.size
        r, g, b = read_bytes(3).unpack('C*')
        palette[i] = (r << 16) | (g << 8) | b
      end
      palette
    end

    def read_app_extension(gif)
      app_label = read_block(11)
      blocks = read_remaining_blocks
      case app_label
      when 'NETSCAPE2.0'
        for block in blocks
          case block.getbyte(0)
          when 0x01
            gif.loop_count = block[1, 2].unpack('v')[0]
          end
        end
      end
    end

    def read_graphic_control_extension(graphic)
      bits, delay_time, transparent_color_index = read_block(4).unpack('CvC')
      graphic.disposal_method = (bits >> 2) & 0x7
      graphic.user_input = (bits & 0x02) == 0x02
      graphic.delay_time = delay_time == 0 ? nil : delay_time / 100.0
      graphic.transparent_color_index = (bits & 0x01) == 0x01 ? transparent_color_index : nil
      read_block(0)
    end

    def read_plain_text_extension(graphic = nil)
      text = Text.new(graphic)
      text.x, text.y, text.width, text.height, text.cell_width, text.cell_height, text.foreground_color_index, text.background_color_index = read_block(12).unpack('vvvvCCCC')
      text.data = read_remaining_blocks.join
      text
    end

    def read_image(graphic = nil)
      image = Image.new(graphic)
      read_image_descriptor(image)
      image.data = read_image_data
      image
    end

    def read_image_descriptor(image)
      image.x, image.y, image.width, image.height, bits = read_bytes(9).unpack('vvvvC')
      image.interlaced = (bits & 0x40) == 0x40
      if (bits & 0x80) == 0x80
        image.palette = read_palette((bits & 0x7) + 1)
        image.palette.sorted = (bits & 0x20) == 0x20
      end
    end

    def read_image_data
      min_code_size = read_byte
      decompressor = Decompressor.new(read_remaining_blocks.join, min_code_size)
      chunks = []
      while chunk = decompressor.read_bytes
        chunks << chunk
      end
      chunks.join
    end

    def read_remaining_blocks
      blocks = []
      loop do
        block_size = read_byte
        break if block_size == 0
        blocks << read_bytes(block_size)
      end
      blocks
    end

    def read_block(size = nil)
      block_size = read_byte
      unless size.nil? || block_size == size
        raise IOError, "unexpected block size (#{block_size} for #{size})"
      end
      read_bytes(block_size)
    end

    def read_byte
      byte = @io.getbyte
      raise EOFError, 'end of GIF' if byte.nil?
      byte
    end

    def read_bytes(size)
      bytes = @io.read(size)
      raise EOFError, 'end of GIF' if bytes.nil? || bytes.size < size
      bytes
    end
  end

  class Writer < Stream
    def self.write_file(path, gif)
      open_file(path, 'wb') { |writer| writer.write(gif) }
    end

    def write(gif)
      writer_header
      write_screen_descriptor(gif)
      write_loop_count(gif.loop_count) unless gif.loop_count.nil?
      for comment in gif.comments
        write_comment(comment)
      end
      for graphic in gif.graphics
        write_graphic(graphic)
      end
      write_footer
    end

    def write_header
      write_bytes('GIF89a')
    end

    def write_screen_descriptor(gif)
      bits = gif.source_color_depth << 4
      unless gif.palette.nil?
        bits |= 0x80
        bits |= (gif.palette.depth - 1) & 0x7
        bits |= 0x08 if gif.palette.sorted
      end
      aspect_byte = gif.aspect_ratio.nil? ? 0 : [[(gif.aspect_ratio * 64.0).floor - 15, 0].max, 255].min
      write_bytes([gif.width, gif.height, bits, gif.background_color_index, aspect_byte].pack('vvCCC'))
      write_palette(gif.palette) unless gif.palette.nil?
    end

    def write_palette(palette)
      for i in 0 ... palette.size
        write_byte(palette[i] >> 16)
        write_byte(palette[i] >> 8)
        write_byte(palette[i])
      end
    end

    def write_loop_count(loop_count)
      write_byte('!'.ord)
      write_byte(0xFF)
      write_block('NETSCAPE2.0')
      write_block([0x01, loop_count].pack('Cv'))
      write_byte(0)
    end

    def write_comment(comment)
      write_byte('!'.ord)
      write_byte(0xFE)
      write_blocks(comment)
    end

    def write_graphic(graphic)
      write_graphic_control_extension(graphic)
      case graphic
      when Image
        write_image(graphic)
      when Text
        write_plain_text_extension(graphic)
      else
        raise TypeError, "unsupported graphic type #{graphic.class}"
      end
    end

    def write_graphic_control_extension(graphic)
      write_byte('!'.ord)
      write_byte(0xF9)
      bits = graphic.disposal_method << 2
      bits |= 0x02 if graphic.user_input
      bits |= 0x01 unless graphic.transparent_color_index.nil?
      delay_time = graphic.delay_time.nil? ? 0 : (grahpic.delay_time * 100.0).floor
      transparent_color_index = graphic.transparent_color_index.nil? ? 0 : graphic.transparent_color_index
      write_block([bits, delay_time, transparent_color_index].pack('CvC'))
      write_byte(0)
    end

    def write_plain_text_extension(text)
      write_byte('!'.ord)
      write_byte(0x01)
      write_block([text.x, text.y, text.width, text.height, text.cell_width, text.cell_height, text.foreground_color_index, text.background_color_index].pack('vvvvCCCC'))
      write_blocks(text.data)
    end

    def write_image(image)
      write_byte(','.ord)
      write_image_descriptor(image)
      write_image_data(image)
    end

    def write_image_descriptor(image)
      bits = 0
      bits |= 0x40 if interlaced
      unless image.palette.nil?
        bits |= 0x80
        bits |= (image.palette.depth - 1) & 0x7
        bits |= 0x20 if image.palette.sorted
      end
      write_bytes([image.x, image.y, image.width, image.height, bits].pack('vvvvC'))
      write_palette(image.palette) unless image.palette.nil?
    end

    def write_image_data(image)
      min_code_size = find_min_code_size(image.data)
      compressor = Compressor.new(min_code_size)
      image.data.each_byte { |b| compressor.write_byte(b) }
      compressor.write_stop
      write_byte(min_code_size)
      write_blocks(compressor.data)
    end

    def find_min_code_size(data)
      max_byte = data.bytes.max
      code_size = 2
      unless max_byte.nil?
        until max_byte < (1 << code_size)
          code_size += 1
        end
      end
      code_size
    end

    def write_footer
      write_byte(';'.ord)
    end

    def write_blocks(bytes)
      0.step(bytes.size - 1, 255) do |i|
        write_block(bytes[i, 255])
      end
      write_byte(0)
    end

    def write_block(bytes)
      write_byte(bytes.size)
      write_bytes(bytes)
    end

    def write_bytes(bytes)
      @io.write(bytes)
    end

    def write_byte(b)
      @io.putc(b)
    end
  end

  class Compressor
    attr_reader :data
    attr_reader :code_size

    def initialize(min_code_size)
      @data = [].pack('C*')
      @bits = 0
      @bit_count = 0
      @min_code_size = min_code_size
      @clear_code = 1 << min_code_size
      clear
    end

    def clear
      write_code(@clear_code)
      @next_code = @clear_code + 2
      @code_size = @min_code_size + 1
      @dict = {}
      for code in 0 ... @clear_code
        @dict[[code].pack('C')] = code
      end
      @seq = [].pack('C*')
    end

    def stop
      write_code(@dict[@seq]) unless @seq.empty?
      write_code(@clear_code + 1)
      flush_bits
    end

    def write_byte(b)
      @seq << b
      unless @dict.include?(@seq)
        write_code(@seq.chop)
        # TODO: clear if maximum code size exceeded
        @dict[@seq] = @next_code
        @code_size += 1 if @next_code == (1 << @code_size)
        @next_code += 1
        @seq = [b].pack('C')
      end
    end

    def write_code(code)
      @bits |= code << @bit_count
      @bit_count += @code_size
      while @bit_count >= 8
        @data << @bits & 0xFF
        @bits >>= 8
        @bit_count -= 8
      end
    end

    def flush_bits
      if @bit_count > 0
        @data << @bits
        @bits = 0
        @bit_count = 0
      end
    end
  end

  class Decompressor
    def initialize(data, min_code_size)
      @data = data
      @index = 0
      @bits = 0
      @bit_count = 0
      @min_code_size = min_code_size
      @clear_code = 1 << min_code_size
      clear
    end

    def clear
      @code_size = @min_code_size + 1
      @next_code = @clear_code + 2
      @last_code = (1 << @code_size) - 1
      @dict = Array.new(@clear_code) { |i| [i].pack('C') }
      @prev_seq = nil
    end

    def read_code
      while @bit_count < @code_size
        @bits |= @data.getbyte(@index) << @bit_count
        @bit_count += 8
        @index += 1
      end
      code = @bits & @last_code
      @bits >>= @code_size
      @bit_count -= @code_size
      code
    end

    def read_bytes
      code = read_code
      return nil if code == @clear_code + 1
      if code == @clear_code
        clear
        code = read_code
      end
      seq = @dict[code]
      unless @prev_seq.nil?
        if seq.nil?
          new_seq = @prev_seq + @prev_seq[0]
          seq = new_seq
        else
          new_seq = @prev_seq + seq[0]
        end
        @dict[@next_code - 1] = new_seq
      end
      # On compression, every time a code is outputted a dictionary entry is added and the next code is incremented.
      # Do all that here as well, except we can't add the dictionary entry yet, until we know the next character.
      # Instead record the sequence so that we can add it at the start of the next iteration.
      @prev_seq = nil
      if @next_code < (1 << 13)
        if @next_code > @last_code && @code_size < 12
          @code_size += 1
          @last_code = (1 << @code_size) - 1
        end
        @next_code += 1
        @prev_seq = seq
      end
      seq
    end
  end
end

if $0 == __FILE__
  def explode_gif(path)
    dir_path = path.chomp('.gif')
    Dir.mkdir(dir_path) unless File.directory?(dir_path)
    gif = GIF::Reader.read_file(path)
    gif.graphics.each_with_index do |graphic, i|
      case graphic
      when GIF::Image
        File.open(File.join(dir_path, sprintf('%d.bmp', i)), 'wb') do |io|
          write_bmp(io, graphic, graphic.palette || gif.palette)
        end
      end
    end
  end

  def write_bmp(io, image, palette)
    line_size = image.width + (-image.width) % 4
    bfh = Array.new(5, 0)
    bfh[0] = 'BM'
    bfh[4] = 14 + 40 + palette.size * 4
    bfh[1] = bfh[4] + line_size * image.height
    io.write(bfh.pack('a2VvvV'))
    bih = Array.new(11, 0)
    bih[0] = 40
    bih[1] = image.width
    bih[2] = -image.height
    bih[3] = 1
    bih[4] = 8
    bih[6] = line_size * image.height
    bih[9] = palette.size
    io.write(bih.pack('VVVvvVVVVVV'))
    io.write(Array.new(palette.size) { |i| palette[i] }.pack('V*'))
    for y in 0 ... image.height
      line = [0].pack('C') * line_size
      for x in 0 ... image.width
        line.setbyte(x, image.color_index(x, y))
      end
      io.write(line)
    end
  end

  for path in ARGV
    explode_gif(path)
  end
end

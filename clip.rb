require 'dl'
require 'dl/import'
require 'dl/types'

module Clipboard
  module Win32API
    include DL::Importer

    def dlload(*libs)
      super
      include DL::Win32Types
      if DL::SIZEOF_VOIDP == DL::SIZEOF_LONG_LONG
        typealias 'SIZE_T', 'unsigned long long'
        typealias 'SSIZE_T', 'long long'
      else
        typealias 'SIZE_T', 'unsigned long'
        typealias 'SSIZE_T', 'long'
      end
    end

    def winapi(signature)
      extern(signature, :stdcall)
    end
  end

  module Kernel32
    extend Win32API
    dlload 'Kernel32'

    GHND = 0x0042

    typealias 'HGLOBAL', 'HANDLE'

    winapi 'HGLOBAL GlobalAlloc(UINT, SIZE_T)'
    winapi 'HGLOBAL GlobalFree(HGLOBAL)'
    winapi 'SIZE_T GlobalSize(HGLOBAL)'
    winapi 'PVOID GlobalLock(HGLOBAL)'
    winapi 'BOOL GlobalUnlock(HGLOBAL)'
  end

  module User32
    extend Win32API
    dlload 'User32'

    CF_UNICODE = 13

    winapi 'BOOL IsClipboardFormatAvailable(UINT)'
    winapi 'BOOL OpenClipboard(HWND)'
    winapi 'BOOL CloseClipboard()'
    winapi 'BOOL EmptyClipboard()'
    winapi 'HANDLE GetClipboardData(UINT)'
    winapi 'HANDLE SetClipboardData(UINT, HANDLE)'
  end

  def self.copy(encoding = Encoding.default_internal)
    return decode(get_data(User32::CF_UNICODE), encoding)
  end

  def self.paste(str)
    return set_data(User32::CF_UNICODE, encode(str))
  end

  def self.encode(str)
    return nil if str.nil?
    wchars = str.encode(Encoding::UTF_16LE).unpack('S<*')
    raise ArgumentError, 'invalid null terminator' if wchars.include?(0)
    wchars << 0
    return wchars.pack('S*')
  end

  def self.decode(data, encoding = Encoding.default_internal)
    return nil if data.nil?
    wchars = data.unpack('S*')
    len = wchars.index(0)
    raise ArgumentError, 'missing null terminator' if len.nil?
    str = wchars[0, len].pack('S<*').force_encoding(Encoding::UTF_16LE)
    return str.encode(encoding || Encoding.default_external)
  end

  def self.get_data(format)
    data = nil
    unless User32::IsClipboardFormatAvailable(format) == 0
      unless User32::OpenClipboard(0) == 0
        hdata = User32::GetClipboardData(format)
        unless hdata == 0
          data = copy_global_data(hdata)
        end
        User32::CloseClipboard()
      end
    end
    return data
  end

  def self.set_data(format, data)
    return false if data.nil?
    hdata = create_global_data(data)
    unless hdata == 0
      unless User32::OpenClipboard(0) == 0
        unless User32::EmptyClipboard() == 0
          unless User32::SetClipboardData(User32::CF_UNICODE, hdata) == 0
            User32::CloseClipboard()
            return true
          end
        end
        User32::CloseClipboard()
      end
      destroy_global_data(hdata)
    end
    return false
  end

  def self.create_global_data(data)
    return 0 if data.empty?
    hdata = Kernel32::GlobalAlloc(Kernel32::GHND, data.bytesize)
    unless hdata == 0
      ptr = Kernel32::GlobalLock(hdata)
      unless ptr.null?
        ptr[0, data.bytesize] = data
        Kernel32::GlobalUnlock(hdata)
        return hdata
      end
      Kernel32::GlobalFree(hdata)
    end
    return 0
  end

  def self.destroy_global_data(hdata)
    Kernel32::GlobalFree(hdata)
  end

  def self.copy_global_data(hdata)
    data = nil
    size = Kernel32::GlobalSize(hdata)
    unless size == 0
      ptr = Kernel32::GlobalLock(hdata)
      unless ptr.null?
        data = ptr[0, size]
        Kernel32::GlobalUnlock(ptr)
      end
    end
    return data
  end
end

if $0 == __FILE__
  case ARGV[0]
  when 'copy'
    str = Clipboard.copy
    if str.nil?
      $stderr.puts("Error copying from clipboard")
      exit(false)
    end
    $stdout.write(str)
  when 'paste'
    str = $stdin.read
    unless Clipboard.paste(str)
      $stderr.puts("Error pasting to clipboard")
      exit(false)
    end
  else
    $stderr.puts("Usage: ruby #{$0} [copy|paste]")
    exit(false)
  end
end

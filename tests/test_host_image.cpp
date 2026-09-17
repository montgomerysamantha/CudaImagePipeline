#include "core/image.hpp"
#include "tests/test_runner.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{
// Keep these host-only tests independent of the CUDA test helpers.
void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void testDefaultImage()
{
    // Verify that a default image describes empty RGB storage.
    // Act
    const HostImage image;

    // Assert
    require(
        image.width == 0 &&
        image.height == 0 &&
        image.channels == 3,
        "Default image metadata changed"
    );

    require(
        image.pixels.empty() && image.view().bytes() == 0,
        "Default image must be empty"
    );
}

void testManualSetupAndViews()
{
    // Existing callers must still be able to assign dimensions and pixels
    // directly. A view refers to that storage; it does not copy the pixels.
    // Arrange
    HostImage image;
    image.width = 2;
    image.height = 1;
    image.channels = 3;
    image.pixels = {10, 20, 30, 40, 50, 60};

    // Act
    auto view = image.view();

    // Assert
    require(
        view.width == 2 && view.height == 1 && view.channels == 3,
        "Mutable view must report manually assigned dimensions"
    );

    require(
        view.bytes() == 6 && view.data == image.pixels.data(),
        "Mutable view must refer to the owned pixel storage"
    );

    // Act: change the second pixel's green channel through the view.
    view.data[4] = 99;

    // Assert
    require(
        image.pixels[4] == 99,
        "View writes must update the image"
    );

    // Act: request a read-only view of the same image.
    const HostImage& readOnly = image;
    const auto constView = readOnly.view();

    // Assert: const views expose the same bytes but prohibit writing through
    // their data pointer. The pointer type is checked at compile time.
    static_assert(
        std::is_same_v<decltype(constView.data), const unsigned char*>
    );

    require(
        constView.width == 2 &&
        constView.height == 1 &&
        constView.channels == 3,
        "Const view dimensions changed"
    );

    require(
        constView.bytes() == 6 &&
        constView.data == image.pixels.data() &&
        constView.data[4] == 99,
        "Const view must expose the current pixels"
    );
}

void testCopyOwnsIndependentPixels()
{
    // Verify that brace initialization still works and copying an image
    // creates independent pixel storage.
    // Arrange
    const HostImage original{2, 1, 3, {10, 20, 30, 40, 50, 60}};

    // Act
    HostImage copy = original;

    // Assert
    require(
        copy.width == original.width &&
        copy.height == original.height &&
        copy.channels == original.channels &&
        copy.pixels == original.pixels,
        "Copy must preserve dimensions and pixels"
    );

    // Act
    copy.view().data[0] = 200;

    // Assert
    require(
        original.pixels[0] == 10,
        "Copy must not share writable storage"
    );
}

void testManualStorageResize()
{
    // Preserve the manual allocation pattern used before allocate() existed.
    // Growing a vector preserves its existing bytes; it does not resample them.
    // Arrange
    HostImage image{1, 1, 3, {10, 20, 30}};

    // Act
    image.width = 2;
    image.height = 2;
    image.pixels.resize(image.view().bytes());

    // Obtain a fresh view because resizing can move the vector's storage.
    const auto view = image.view();

    // Assert: 2 x 2 pixels with 3 channels need 12 bytes.
    require(
        view.bytes() == 12 && image.pixels.size() == 12,
        "Manual allocation must use the updated shape"
    );

    require(
        view.data == image.pixels.data(),
        "Fresh view must use current storage"
    );

    require(
        image.pixels[0] == 10 &&
        image.pixels[1] == 20 &&
        image.pixels[2] == 30,
        "Growing pixel storage must preserve existing bytes"
    );
}

void testAllocateSetsShapeAndStorage()
{
    // Verify that allocate() sets the dimensions and creates writable storage.
    // Arrange
    HostImage image;

    // Act
    image.allocate(5, 2, 3);

    // Assert: 5 x 2 RGB pixels need 30 bytes.
    require(
        image.width == 5 && image.height == 2 && image.channels == 3,
        "Allocation must set the requested shape"
    );

    require(
        image.pixels.size() == 30 && image.view().bytes() == 30,
        "Allocation must provide storage for every channel of every pixel"
    );

    const bool allPixelsAreZero = std::all_of(
        image.pixels.begin(),
        image.pixels.end(),
        [](unsigned char value)
        {
            return value == 0;
        }
    );

    require(
        allPixelsAreZero,
        "New pixel bytes must be zero initialized"
    );

    // Act: write to the last allocated byte through a view.
    image.view().data[29] = 123;

    // Assert
    require(
        image.pixels.back() == 123,
        "Allocated storage must be writable through a view"
    );
}

void testRepeatedAllocation()
{
    // allocate() uses vector resize semantics: retain existing bytes,
    // zero new bytes when growing, and remove trailing bytes when shrinking.
    // Arrange
    HostImage image{1, 1, 3, {10, 20, 30}};
    const std::vector<unsigned char> expectedSameSize{10, 20, 30};
    const std::vector<unsigned char> expectedLarger{10, 20, 30, 0, 0, 0};
    const std::vector<unsigned char> expectedSmaller{10, 20};

    // Act: allocate the same shape again.
    image.allocate(1, 1, 3);

    // Assert
    require(
        image.pixels == expectedSameSize,
        "Same-size allocation must preserve bytes"
    );

    // Act: grow from one RGB pixel to two.
    image.allocate(2, 1, 3);

    // Assert
    require(
        image.width == 2 &&
        image.height == 1 &&
        image.channels == 3 &&
        image.pixels == expectedLarger,
        "Growing must preserve the prefix and zero initialize new bytes"
    );

    // Act: shrink to two single-channel pixels. This changes storage and
    // metadata only; it does not perform an RGB-to-grayscale conversion.
    image.allocate(1, 2, 1);

    // Assert
    require(
        image.width == 1 &&
        image.height == 2 &&
        image.channels == 1 &&
        image.pixels == expectedSmaller,
        "Shrinking and changing channels must update shape and storage"
    );
}

template <typename Exception>
void requireRejectedWithoutChanges(int width, int height, int channels)
{
    // Shared check for invalid requests: the expected exception must be
    // thrown, and the existing image must remain usable and unchanged.
    // Arrange
    HostImage image{2, 1, 3, {10, 20, 30, 40, 50, 60}};
    const HostImage before = image;
    bool rejected = false;

    // Act
    try
    {
        image.allocate(width, height, channels);
    }
    catch (const Exception&)
    {
        rejected = true;
    }

    // Assert: a different exception type propagates and also fails the test.
    require(
        rejected,
        "Invalid allocation must throw the expected exception"
    );

    require(
        image.width == before.width &&
        image.height == before.height &&
        image.channels == before.channels &&
        image.pixels == before.pixels,
        "Rejected allocation must preserve the previous image"
    );
}

void testInvalidAllocationPreservesImage()
{
    // Check each argument independently while keeping the other two valid.
    // Include zero, a negative value, and the smallest representable int.
    for (int invalid : {0, -1, std::numeric_limits<int>::min()})
    {
        requireRejectedWithoutChanges<std::invalid_argument>(invalid, 2, 3);
        requireRejectedWithoutChanges<std::invalid_argument>(2, invalid, 3);
        requireRejectedWithoutChanges<std::invalid_argument>(2, 2, invalid);
    }
}

void testOversizedAllocationPreservesImage()
{
    // This product exceeds size_t on supported 32/64-bit targets.
    // Reject it before attempting an enormous allocation or wrapping the size.
    // Arrange
    const int largest = std::numeric_limits<int>::max();

    // Act and assert
    requireRejectedWithoutChanges<std::length_error>(largest, largest, largest);
}
}

int main()
{
    try
    {
        test::Runner runner;

        runner.run("Default image", testDefaultImage);
        runner.run("Manual setup and views", testManualSetupAndViews);
        runner.run("Independent copy", testCopyOwnsIndependentPixels);
        runner.run("Manual storage resize", testManualStorageResize);
        runner.run("Allocation shape and storage", testAllocateSetsShapeAndStorage);
        runner.run("Repeated allocation", testRepeatedAllocation);
        runner.run("Invalid allocation preserves image", testInvalidAllocationPreservesImage);
        runner.run("Oversized allocation preserves image", testOversizedAllocationPreservesImage);

        runner.summary("HostImage");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "HostImage test failed: " << error.what() << '\n';
        return 1;
    }
}

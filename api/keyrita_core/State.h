#pragma once

namespace kc
{
/**
 * @brief       A notification that gets send when internal state is modified.
 */
class StateChangeNotification
{
};

/**
 * @brief       Defines the contract that must be implemented for any read write state
 */
class IReadWriteState
{
};
}   // namespace kc
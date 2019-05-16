/**
This file contains the definition of the exception that can be thrown
by the hcrf library
**/

#ifndef HCRFEXCEP_H
#define HCRFEXCEP_H
#include <stdexcept>

// Thrown when a null pointer is detected in an unexpected place
class HcrfBadPointer : public std::runtime_error
{
public:
    explicit HcrfBadPointer(const std::string& what_arg):
    runtime_error(what_arg) {}
};

// Thrown for feature not implemented
class HcrfNotImplemented : public std::logic_error
{
public:
    explicit HcrfNotImplemented(const std::string& what_arg) :
    logic_error(what_arg) {}
};

class HcrfBadModel : public std::invalid_argument
{
  public:
    explicit HcrfBadModel(const std::string& what_arg) :
    invalid_argument(what_arg) {}
};

// Thrown for invalid file name
class BadFileName : public std::invalid_argument
{
public:
    explicit BadFileName(const std::string& what_arg) :
    invalid_argument(what_arg) {}
};

class BadIndex : public std::invalid_argument
{
public:
    explicit BadIndex(const std::string& what_arg) :
    invalid_argument(what_arg) {}
};

class InvalidOptimizer : public std::invalid_argument
{
  public:
    explicit InvalidOptimizer(const std::string& what_arg):
    invalid_argument(what_arg) {}
};

#endif
